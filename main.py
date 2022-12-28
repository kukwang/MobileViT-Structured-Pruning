import torch
from models import create_models

def main():
    img = torch.randn(5, 3, 256, 256)

    vit = mobilevit_xxs()
    out = vit(img)
    print(out.shape)
    print(count_parameters(vit))

    vit = mobilevit_xs()
    out = vit(img)
    print(out.shape)
    print(count_parameters(vit))

    vit = mobilevit_s()
    out = vit(img)
    print(out.shape)
    print(count_parameters(vit))
    return

if __name__ == '__main__':
    main()