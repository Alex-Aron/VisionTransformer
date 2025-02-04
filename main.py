import torch
from my_vit_models import MyViT2D, MyViT3D, MyViT4D

def main():
#    device = torch.device("cuda")
 #   x = torch.rand(8, 3, 224, 224)
  #  model = MyViT2D()
   # pred = model(x)
   # assert x.shape == pred.shape #Ensure out and input shape is the same as specified
   # loss = torch.sum(pred)
   # loss.backward()
   # print(f'2D Loss computed: {loss.item()}')
    x = torch.rand(8, 3, 224, 224, 32)
    model = MyViT3D()
    pred = model(x)
    assert x.shape == pred.shape
    loss = torch.sum(pred)
    loss.backward()
    print(f'3D Loss computed: {loss.item()}')
    x = torch.rand(2, 3, 224, 224, 32, 4)
    #Changed tensor size from 8 to 2 because it was giving CUDA out of memory error. 
   # model = MyViT4D()
   # pred = model(x)
   # assert x.shape == pred.shape
   # loss = torch.sum(pred)
   # loss.backward()
   # print(f'4D Loss computed: {loss.item()}')

if __name__ == '__main__':
    main()