import core
import neuro_network
import torch
import torchvision


model = core.Model(model=neuro_network.Simple(), params_path="./.core/model/model.pt")

model.load_trans(torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                 torchvision.transforms.Normalize(mean=[0.5], std=[0.5]),
                                                 torchvision.transforms.Resize((64, 64), antialias=True)
                                                 ]))

model.load_train_data_set(torchvision.datasets.FashionMNIST(root=".download/FashionMNIST",
                                                            train=True,
                                                            transform=model.transform,
                                                            download=True))

model.load_test_data_set(torchvision.datasets.FashionMNIST(root=".download/FashionMNIST",
                                                           train=False,
                                                           transform=model.transform,
                                                           download=True))

model.train(8, 1e-4, batch_sizes=256)

model.score()

model.save_params(path="./.core/model/model.pt")
