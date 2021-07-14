import numpy as np
import warnings

try:
    from PIL import Image
    import torch
    import torchvision
except ImportError:
    warnings.warn("Pillow, torch or torchvision not installed! Image2Vector will fail!")


class Image2vector():
    def __init__(self, model='resnet', use_gpu=True):
        if model == 'resnet':
            import torch
            from torchvision.models import resnet50

            def embedding(model, x):
                x = model.conv1(x)
                x = model.bn1(x)
                x = model.relu(x)
                x = model.maxpool(x)

                x = model.layer1(x)
                x = model.layer2(x)
                x = model.layer3(x)
                x = model.layer4(x)
                x = model.avgpool(x)
                x = torch.flatten(x, 1)

                return x

            self.model = resnet50(pretrained=True)
            self.embedding = embedding
        elif model == 'vgg':
            from torchvision.models import vgg19
            self.model = vgg19(pretrained=True)
            raise NotImplementedError()

        self.device = 'cuda' if use_gpu else 'cpu'

    def predict(self, images):
        """
        :param images: numpy array
        :return: numpy array of shape (n_samples,embedding_size)
        """
        self.model.to(self.device)

        import torch
        from torch import Tensor
        from torch.utils.data import DataLoader, TensorDataset

        images = Tensor(images).to(self.device)
        image_loader = DataLoader(
            TensorDataset(images), batch_size=128, shuffle=False)

        embeddings = None
        with torch.no_grad():
            for i, data in enumerate(image_loader):
                batch_x = data[0]
                logits = self.embedding(self.model, batch_x)
                if embeddings is None:
                    embeddings = logits.to('cpu').detach().numpy()
                else:
                    embeddings = np.concatenate((embeddings, logits.to('cpu').detach().numpy()), 0)

        return embeddings
