from torch.utils.data import DataLoader

inputs = torch.randn(64, 8)
outputs = torch.randn(64, 3)

dataset = MyDataset(inputs, outputs)

loader = DataLoader(dataset, batch_size = 2, num_workers = 0)



