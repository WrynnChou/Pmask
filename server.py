
import models, torch


class Server(object):
	
	def __init__(self, conf, eval_dataset):
	
		self.conf = conf 
		
		self.global_model = models.get_model(self.conf["model_name"]) 
		
		self.eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=self.conf["batch_size"], shuffle=True)
		
	
	def model_aggregate(self, weight_accumulator,e,nonzero_average):
		warmup = self.conf["warmup"]
		if e < warmup:
			CompressionMethod = "Full"
		else:
			CompressionMethod = self.conf["Compression_method"]

		if CompressionMethod == "Permutation Mask":
			weight = 1
			for name, data in self.global_model.state_dict().items():
				if data.shape != torch.Size([]):
					update_per_layer = weight_accumulator[name] * weight
				else:
					update_per_layer = weight_accumulator[name] * self.conf["lambda"]
				#update_per_layer 是各个client更新后的权重*(1/k)
				if data.type() != update_per_layer.type():
					data.add_(update_per_layer.to(torch.int64))
				else:
					data.add_(update_per_layer)


		elif CompressionMethod == "Permutation Top-1 Mask":
			for name, data in self.global_model.state_dict().items():
				if data.shape != torch.Size([]):
					update_per_layer = weight_accumulator[name] /(nonzero_average[name])
				else:
					update_per_layer = weight_accumulator[name] * self.conf["lambda"]
				if data.type() != update_per_layer.type():
					data.add_(update_per_layer.to(torch.int64))
				else:
					data.add_(update_per_layer)
		else:
			weight = self.conf["lambda"]
			for name, data in self.global_model.state_dict().items():
				if data.shape != torch.Size([]):
					update_per_layer = weight_accumulator[name] * weight
				else:
					update_per_layer = weight_accumulator[name] * self.conf["lambda"]
				#update_per_layer 是各个client更新后的权重*(1/k)
				if data.type() != update_per_layer.type():
					data.add_(update_per_layer.to(torch.int64))
				else:
					data.add_(update_per_layer)
				
	def model_eval(self):
		self.global_model.eval()
		
		total_loss = 0.0
		correct = 0
		dataset_size = 0
		for batch_id, batch in enumerate(self.eval_loader):
			data, target = batch 
			dataset_size += data.size()[0]
			
			if torch.cuda.is_available():
				data = data.cuda()
				target = target.cuda()
				
			
			output = self.global_model(data)
			
			total_loss += torch.nn.functional.cross_entropy(output, target,
											  reduction='sum').item() # sum up batch loss
			pred = output.data.max(1)[1]  # get the index of the max log-probability
			correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

		acc = 100.0 * (float(correct) / float(dataset_size))
		total_l = total_loss / dataset_size

		return acc, total_l