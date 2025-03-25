import argparse, json
import torch, random
import numpy as np
from server import *
from client import *
import models, datasets
import time
import datetime

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Federated Learning')
	parser.add_argument('-c', '--conf', dest='conf')
	args = parser.parse_args()
	

	with open(args.conf, 'r') as f:
		conf = json.load(f)	
	
	
	train_datasets, eval_datasets = datasets.get_dataset("./data/", conf["type"])
	
	server = Server(conf, eval_datasets)
	clients = []
	t0 = time.time()
	for c in range(conf["no_models"]):
		clients.append(Client(conf, server.global_model, train_datasets, c, {}))

	for c in clients:
		# Local——gradient获取结构
		for name, params in server.global_model.state_dict().items():
			c.local_accumulated_gradient[name] = torch.zeros_like(params)

	print("\n\n")
	for e in range(conf["global_epochs"]):

		candidates = random.sample(clients, conf["k"])
		warmup = conf["warmup"]
		lamb = conf["lambda"]
		rho = conf["rho"]
		weight_accumulator = {}#清空聚合器
		nonzero_average = {} #记录非零均值的个数
		up = {}
		down = {}
		up_v = {}
		down_v = {}
		ratio_tensor = {}
		for name, params in server.global_model.state_dict().items():
			weight_accumulator[name] = torch.zeros_like(params)
			nonzero_average[name] = torch.ones_like(params)
			up[name] = torch.zeros_like(params)
			up_v[name] = 0
			down_v[name] = 0
			ratio_tensor[name] = 0
			down[name] = torch.zeros_like(params)
		if e < warmup:
			CompressionMethod = "Full"
		else:
			CompressionMethod = conf["Compression_method"]
			CompressionRate = conf["Compression_rate"]
		step = conf["step"]


		if CompressionMethod == "Permutation Mask" or CompressionMethod == "Permutation Top-1 Mask":
			permutaion_uni = {}
			for name, params in server.global_model.state_dict().items():
				if params.shape != torch.Size([]):
					uni_mask_perm = torch.randn_like(params)
					permutaion_uni[name] = uni_mask_perm
				else:
					permutaion_uni[name] = params

		for c in candidates:

			#print(c.client_id)

			if CompressionMethod == "Random Mask":
				#print(CompressionMethod)
				diff = c.local_train(server.global_model)
				diff_masked = {}
				for name, params in server.global_model.state_dict().items():
					if params.shape != torch.Size([]):
						gt = c.local_accumulated_gradient[name]*step + diff[name]
						uni_mask = torch.randn_like(params)
						threshold = np.percentile(uni_mask.cpu(), CompressionRate)
						mask = torch.gt(uni_mask, threshold)
						diff_masked[name] = torch.masked_fill(gt, mask, 0)
						c.local_accumulated_gradient[name] = torch.masked_fill(gt, ~mask, 0)
						Clipping_Value = torch.max(abs(diff[name]))
						clip = torch.gt(abs(c.local_accumulated_gradient[name]),Clipping_Value)
						c.local_accumulated_gradient[name] = torch.masked_fill(c.local_accumulated_gradient[name],clip,0)
					else:
						diff_masked[name] = diff[name]
					weight_accumulator[name].add_(diff_masked[name])


			elif CompressionMethod == "Full":
				#print(CompressionMethod)
				diff = c.local_train(server.global_model)
				for name, params in server.global_model.state_dict().items():
					weight_accumulator[name].add_(diff[name])


			elif CompressionMethod == "TopK":
				#print(CompressionMethod)
				diff = c.local_train(server.global_model)
				diff_masked = {}
				for name, params in server.global_model.state_dict().items():
					if params.shape != torch.Size([]):
						gt = c.local_accumulated_gradient[name]*step + diff[name]
						threshold = np.percentile(abs(gt).cpu(), 100-CompressionRate)
						mask = torch.lt(abs(gt), threshold)
						diff_masked[name] = torch.masked_fill(gt, mask, 0)
						temp = torch.masked_fill(gt, ~mask, 0)
						Clipping_Value = torch.max(0.5*abs(diff[name]))
						clip = torch.gt(abs(temp), Clipping_Value)
						c.local_accumulated_gradient[name] = torch.masked_fill(temp,clip,0)
					else:
						diff_masked[name] = diff[name]
					weight_accumulator[name].add_(diff_masked[name])
			elif CompressionMethod == "Permutation Mask":
				#print(CompressionMethod)
				diff = c.local_train(server.global_model)
				diff_masked = {}
				for name, params in server.global_model.state_dict().items():
					if params.shape != torch.Size([]):
						gt = c.local_accumulated_gradient[name]*step + diff[name]
						uni_mask = permutaion_uni[name]
						threshold_1 = np.percentile(uni_mask.cpu(), (CompressionRate*(c.client_id)))
						threshold_2 = np.percentile(uni_mask.cpu(), (CompressionRate*(c.client_id+1)))
						mask_1 = torch.gt(uni_mask, threshold_1)
						mask_2 = torch.gt(uni_mask, threshold_2)
						mask = (~mask_1)|mask_2
						diff_masked[name] = torch.masked_fill(gt, mask, 0)

						c.local_accumulated_gradient[name] = torch.masked_fill(gt, ~mask, 0)
						Clipping_Value = torch.max(abs(diff[name]))
						clip = torch.gt(abs(c.local_accumulated_gradient[name]), Clipping_Value)
						c.local_accumulated_gradient[name] = torch.masked_fill(c.local_accumulated_gradient[name], clip,0)
					else:
						diff_masked[name] = diff[name]

					weight_accumulator[name].add_(diff_masked[name])


			elif CompressionMethod == "Permutation Top-1 Mask":
				#print(CompressionMethod)
				diff = c.local_train(server.global_model)
				diff_masked = {}
				k_1 = 100
				k_2 = 100
				for name, params in server.global_model.state_dict().items():
					if params.shape != torch.Size([]):
						gt = c.local_accumulated_gradient[name]*step + diff[name]
						uni_mask = permutaion_uni[name]
						threshold_1 = np.percentile(uni_mask.cpu(), ((1 - rho)*CompressionRate*(c.client_id)))
						threshold_2 = np.percentile(uni_mask.cpu(), ((1 - rho)*CompressionRate*(c.client_id+1)))
						mask_1 = torch.gt(uni_mask, threshold_1)
						mask_2 = torch.gt(uni_mask, threshold_2)
						mask1 = (~mask_1)|mask_2
						new_para = torch.masked_fill(diff[name], ~mask1, 0)
						threshold = np.percentile(abs(new_para.cpu()), 100 - rho*CompressionRate)
						mask2 = torch.lt(abs(new_para), threshold)
						mask = mask1&mask2
						if c != 0:
							nonzero_average[name].add_((~mask))
						#diff_masked[name] = (torch.masked_fill(gt, mask1, 0)+0.1*torch.masked_fill(gt, mask2, 0))
						threshold_fake = np.percentile(abs(params.cpu()), 100 - CompressionRate)
						mask_fake_1 = torch.lt(abs(new_para), threshold_fake)
						threshold3 = np.percentile(abs(new_para.cpu()), 100 - CompressionRate)
						mask3 = torch.lt(abs(new_para), threshold3)
						diff_masked[name] = torch.masked_fill(gt, mask, 0)
						u1 = torch.sum(torch.square(gt[~mask2]))
						u2 = torch.sum(torch.mul(up[name], torch.masked_fill(gt, mask2, 0)))
						up_v[name] += u1 + u2
						k_1 = min(k_1, ((u1 + 2 * u2 ) / (u1 + u2)))
						d1 = torch.sum(torch.square(gt[mask3]))
						d2 = torch.sum(torch.mul(down[name], torch.masked_fill(gt, ~mask3, 0)))
						down_v[name] += d1 + d2
						k_2 = min(k_2, ((d1 + 2 * d2) / (d1 + d2)))
						up[name].add_(torch.masked_fill(gt, mask2, 0))
						down[name].add_(torch.masked_fill(gt, ~mask3, 0))
						c.local_accumulated_gradient[name] = torch.masked_fill(gt, ~mask, 0)
						Clipping_Value = torch.max(abs(diff[name]))
						clip = torch.gt(abs(c.local_accumulated_gradient[name]), Clipping_Value)
						c.local_accumulated_gradient[name] = torch.masked_fill(c.local_accumulated_gradient[name], clip, 0)
					else:
						diff_masked[name] = diff[name]
					weight_accumulator[name].add_(diff_masked[name])

		for name in server.global_model.state_dict():
			ratio_tensor[name] = up_v[name]/(down_v[name])
		ratio1 = max(ratio_tensor.values())
		ratio2 = min(ratio_tensor.values())
		ratio3 = (sum(ratio_tensor.values()) / len(ratio_tensor))
		server.model_aggregate(weight_accumulator, e, nonzero_average)
		torch.save(server.global_model.state_dict(), '/content/gdrive/MyDrive/Colab Notebooks/fl/train.pth')
		acc, loss = server.model_eval()

		print("Epoch %d, %f, %f, %f, %f, %f, $f, %f \n" % (e, acc, loss, ratio1, ratio2, ratio3, k_1, k_2))

			
		
		
	
		
