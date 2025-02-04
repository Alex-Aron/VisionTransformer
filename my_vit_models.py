import torch
import torch.nn as nn

#Plan:
#Create classes in nn.Module in the order of:
#Patch embedding layer (needed as described in ViT paper)
#Positional embedding layer(applicable to transformers generally)
#----Change: Combined these two layers into one. Created a seperate embedding class for each dimensionality
#Multi-head attention layer
#MLP/Feed forward layer
#Use nn.LayerNorm for normalization
#Create ViT model combining all of these components

class PatchEmbedding2D(nn.Module):
    def __init__(self, image_width, image_height, in_channels, patch_size, embed_dim):
        super(PatchEmbedding2D, self).__init__()
        image_size = image_width * image_height
        assert image_size % patch_size == 0
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = image_width//patch_size * image_height//patch_size
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        #Model will learn positional embeddings as opposed to formula in 'attention is all you need'            
        
        
    def forward(self, x):
        #x is initially the form of the 2d image (Batch, # channels, height, width )
        x = self.proj(x)# Does the projection, making it (Batch, embeddings, height/p, width/p)
        x = x.flatten(2)#Flattens it to (batch, embeddings, h*w/p^2) then swaps 1 and 2
        x = x.transpose(1,2)
        # to (batch, h*w/p^2, embeddings)
        return x + self.pos_embed

class PatchEmbedding3D(nn.Module):
    def __init__(self, image_size, in_channels, patch_size, depth_size, embed_dim):
        super(PatchEmbedding3D, self).__init__()
        
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=image_size, stride=patch_size)
        self.num_patches = (image_size[0] // patch_size[0]) * (image_size[1]//patch_size[1]) * (image_size[2]//patch_size[2])
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

    def forward(self, x):
        #x is (Batch, # channels, height, width, depth)
        print(x.shape)
        x = self.proj(x) # Projection makes it (Batch, embed_dim, height/p, width/p, depth/p)
        print(x.shape)
        print(self.num_patches)
        print(self.pos_embed.shape)
        x = x.flatten(2) # Data flattened to (Batch, embed_dim, h*w*d/p^3=num_patches)
        x = x.transpose(1,2) #Reorder to (batch, num_patches, embed_dim)
        return x + self.pos_embed    

class PatchEmbedding4D(nn.Module):
    def __init__(self, image_size, in_channels, patch_size, depth_size, time_size, embed_dim):
        super(PatchEmbedding4D, self).__init__()
        self.embed_dim = embed_dim
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size[0:3], stride=patch_size[0:3])
        self.temporal_proj = nn.Conv1d(embed_dim, embed_dim, kernel_size=patch_size[3], stride=patch_size[3])
        self.num_patches = (image_size[0] // patch_size[0]) ** 2 * (image_size[2] // patch_size[2]) * (image_size[3] // patch_size[3])
        self.spatial_patches = (image_size[0] // patch_size[0]) ** 2 * (image_size[2] // patch_size[2])
        self.temporal_patches = image_size[3] // patch_size[3]
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
                
    def forward(self, x):
        #Have to get from 
        #(8,3,224,224,32,4)to #(8, 3136, 768)
        print(x.shape)
        B, C, H, W, D, T = x.shape
        # Reshape to handle 3D convolution
        x = x.permute(0, 4, 5, 2, 3, 1)  # (B, C, T, H, W, D)
        print(x.shape)
        x = x.reshape(B * T, C, H, W, D)  # (B*T, C, H, W, D)
        print(x.shape)
        # Apply 3D spatial projection
        x = self.proj(x)  # (B*T, embed_dim, H', W', D')
        print(x.shape)
        # Prepare for temporal projection
        x = x.reshape(B, T, -1, self.spatial_patches)
        print(x.shape)
        x = x.permute(0, 2, 1, 3)  # (B, embed_dim, T, num_patches_spatial)
        print(x.shape)
        x = x.reshape(B * self.spatial_patches, 
                     -1, T)  # (B*num_patches_spatial, embed_dim, T)
        print(x.shape)
        #Apply temporal projection
        x = self.temporal_proj(x)  #(B*num_patches_spatial, embed_dim, T')
        
        # Reshape to final format
        x = x.reshape(B, -1, 2, self.embed_dim)
        x = x.reshape(B, -1, self.embed_dim)  #(B, num_patches_total, embed_dim)
        return x + self.pos_embed
    
class AttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(AttentionLayer, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0
        
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        
        
    def forward(self, x):
        B, N, C = x.shape
        #Create matrices of queries, keys and values
        Q = self.query(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1,2)
        
        #Perform dot product of keys and queries and apply softmax to get attention weights
        attention = (Q @ K.transpose(-2,-1)) / (self.head_dim ** 0.5)
        attention = nn.functional.softmax(attention, dim=-1)
        #Multiply values by the attention and reshape for output.
        x = (attention @ V).transpose(1,2).reshape(B, N, C)
        return self.out(x)        
        
class MLPLayer(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super(MLPLayer, self).__init__()
        self.ff1 = nn.Linear(embed_dim, hidden_dim)
        self.ff2 = nn.Linear(hidden_dim, embed_dim)
    
    def forward(self, x):
        x = self.ff1(x)
        x = nn.functional.gelu(x) #'The MLP contains two layers with a GELU non-linearity' - The paper
        x = self.ff2(x)
        return x    
    
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.attention = AttentionLayer(embed_dim, num_heads)
        self.feedforward = MLPLayer(embed_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = self.dropout(x) #Add dropout 
        x = x + self.feedforward(self.norm2(x))
        return x    


class MyViT2D(nn.Module):
    def __init__(self, in_channels=3, embed_dim=768, image_height = 224, image_width = 224, num_heads=8, hidden_dim=2048, patch_size=16, layers=12):
        super(MyViT2D, self).__init__()
        self.patch_embed = PatchEmbedding2D(image_width, image_height, in_channels, patch_size, embed_dim)
        self.blocks = nn.Sequential(
            *[TransformerEncoder(embed_dim, num_heads, hidden_dim) for _ in range(layers)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, in_channels * patch_size * patch_size)
        
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = self.head(x)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x

class MyViT3D(nn.Module):
    def __init__(self, in_channels=3, embed_dim=768, image_size = (224, 224, 32), num_heads=8, hidden_dim=2048, patch_size=(16, 16, 4), depth_size=4,layers=12):
        super(MyViT3D, self).__init__()
        self.patch_embed = PatchEmbedding3D(image_size, in_channels, patch_size, depth_size ,embed_dim)
        self.blocks = nn.Sequential(
            *[TransformerEncoder(embed_dim, num_heads, hidden_dim) for _ in range(layers)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, in_channels * patch_size[0] * patch_size[1] * patch_size[2])
        
    def forward(self, x):
        print(x.shape)
        B, C, H, W, D = x.shape
        x = self.patch_embed(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = self.head(x)
        x = x.transpose(1, 2).reshape(B, C, H, W, D)
        return x

class MyViT4D(nn.Module):
    def __init__(self, in_channels=3, embed_dim=768, image_size = (224,224,32,4), num_heads=8, hidden_dim=2048, patch_size=(16,16,4,2), depth_size=4, time_size = 2, layers=12):
        super(MyViT4D, self).__init__()
        self.patch_embed = PatchEmbedding4D(image_size, in_channels, patch_size, depth_size, time_size ,embed_dim)
        self.blocks = nn.Sequential(
            *[TransformerEncoder(embed_dim, num_heads, hidden_dim) for _ in range(layers)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, in_channels * patch_size[0] * patch_size[1] * patch_size[2] * patch_size[3])
        
    def forward(self, x):
        B, C, H, W, D, T = x.shape
        x = self.patch_embed(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = self.head(x)
        x = x.transpose(1, 2).reshape(B, C, H, W, D, T)
        return x    

class MyViT3D_V2(nn.Module):
    def __init__(self, in_channels=3, embed_dim=768, image_size = (224,224,32,4), num_heads=8, hidden_dim=2048, patch_size=(16,16,4,2), depth_size=4, time_size = 2, layers=12):
        super(MyViT3D_V2, self).__init__()
        self.patch_embed = PatchEmbedding4D(image_size, in_channels, patch_size, depth_size, time_size ,embed_dim)
        self.blocks = nn.Sequential(
            *[TransformerEncoder(embed_dim, num_heads, hidden_dim) for _ in range(layers)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, in_channels * patch_size[0] * patch_size[1] * patch_size[2] * patch_size[3])
        
    def forward(self, x):
        B, C, H, W, D, T = x.shape
        x = self.patch_embed(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = self.head(x)
        x = x.transpose(1, 2).reshape(B, C, H, W, D, T)
        return x    
