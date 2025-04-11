import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.nn import MultiheadAttention
from tqdm import tqdm
import torch.nn.functional as F

class Attention(nn.Module):
    '''
    Attention Module is used to perform self-attention operation allowing the
    model to attend information from different representation subspaces on an input
    sequence of embeddings.

    Args:
        embed_dim: Dimension size of the hidden embedding
        heads: Number of parallel attention heads

    Methods:
        forward(inp) :-
        Performs the self-attention operation on the input sequence embedding.
        Returns the output of self-attention can be seen as an attention map
        inp (batch_size, seq_len, embed_dim)
        out:(batch_size, seq_len, embed_dim)

    Examples:
        >>> attention = Attention(embed_dim, heads)
        >>> out = attention(inp)
    '''
    def __init__(self, heads, embed_dim):
        super(Attention, self).__init__()
        assert embed_dim % heads == 0, "Embedding dimension must be divisible by number of heads"
        
        self.embed_dim = embed_dim
        self.heads = heads
        self.head_dim = embed_dim // heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, inp):
        batch_size, seq_len, embed_dim = inp.size()

        Q = self.query(inp).view(batch_size, seq_len, self.heads, self.head_dim)
        K = self.key(inp).view(batch_size, seq_len, self.heads, self.head_dim)
        V = self.value(inp).view(batch_size, seq_len, self.heads, self.head_dim)

        Q = Q.transpose(1, 2)  # (batch_size, heads, seq_len, head_dim)
        K = K.transpose(1, 2)  # (batch_size, heads, seq_len, head_dim)
        V = V.transpose(1, 2)  # (batch_size, heads, seq_len, head_dim)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (batch_size, heads, seq_len, seq_len)
        attn = F.softmax(scores, dim=-1)  # (batch_size, heads, seq_len, seq_len)

        context = torch.matmul(attn, V)  # (batch_size, heads, seq_len, head_dim)
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, embed_dim)
      
        out = self.out(context)  
        return out

    



class TransformerBlock(nn.Module):
    '''
    Transformer Block combines both the attention module and the feed-forward
    module with layer normalization, dropout and residual connections. The sequence
    of operations is as follows :-
    Inp -> LayerNorm1 -> Attention -> Residual -> LayerNorm2 -> FeedForward -> Out
    | | | |
    |-------------Addition--------------| |---------------Addition------------|
    Args:
    embed_dim: Dimension size of the hidden embedding
    HCMUS - VNUHCM / FIT / Computer Vision & Cognitive Cybernetics Department
    Advanced Computer Vision - LQN
    Advanced Computer Vision - LQN 3
    heads: Number of parallel attention heads (Default=8)
    mlp_dim: The higher dimension is used to transform the input embedding
    and then resized back to embedding dimension to capture richer information.
    dropout: Dropout value for the layer on attention_scores (Default=0.1)
    Methods:
    forward(inp) :-
    Applies the sequence of operations mentioned above.
    (batch_size, seq_len, embed_dim) -> (batch_size, seq_len, embed_dim)
    Examples:
    >> TB = TransformerBlock(embed_dim, mlp_dim, heads, activation, dropout)
    >> out = TB(inp)
    '''
    def __init__(self, embed_dim, mlp_dim, heads, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiheadAttention(embed_dim, heads)
        self.fc1 = nn.Linear(embed_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, embed_dim)
        self.activation = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
    def forward(self, inp):
        x = self.norm1(inp)
        
        x_transposed = x.transpose(0, 1)  # (seq_len, batch_size, embed_dim)
        attn_out, _ = self.attention(x_transposed, x_transposed, x_transposed)
        attn_out = attn_out.transpose(0, 1)  # Back to (batch_size, seq_len, embed_dim)


        inp = inp + attn_out

        x = self.norm2(inp)

        ff_out = self.fc2(self.dropout2(self.activation(self.fc1(x))))

        out = inp + ff_out

        return out


class Transformer(nn.Module):
    '''
    Transformer combines multiple layers of Transformer Blocks in a sequential
    manner. The sequence
    of the operations is as follows -
    Input -> TB1 -> TB2 -> .......... -> TBn (n being the number of layers) ->
    Output
    Args:
    embed_dim: Dimension size of the hidden embedding in the TransfomerBlock
    mlp_dim: Dimension size of MLP layer in the TransfomerBlock
    layers: Number of Transformer Blocks in the Transformer
    heads: Number of parallel attention heads (Default=8)
    dropout: Dropout value for the layer on attention_scores (Default=0.1)
    Methods:
    forward(inp) :-
    Applies the sequence of operations mentioned above.
    (batch_size, seq_len, embed_dim) -> (batch_size, seq_len, embed_dim)
    '''
    def __init__(self, embed_dim, layers, mlp_dim, heads=8, dropout=0.1):
        super(Transformer, self).__init__()
        self.embed_dim = embed_dim
        self.trans_blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, mlp_dim, heads, dropout)
             for i in range(layers)]
        )

    def forward(self, inp):
        # inp: (batch_size, seq_len, embed_dim)
        for layer in self.trans_blocks:
            inp = layer(inp)
        # inp: (batch_size, seq_len, embed_dim)
        # Return the output of the last transformer block
        return inp


    
class ClassificationHead(nn.Module):
    '''
    Classification Head attached to the first sequence token which is used as
    the arbitrary classification token and used to optimize the transformer model 
    by applying Cross-Entropy loss.

    The sequence of operations is as follows:
    Input -> FC1 -> GELU -> Dropout -> FC2 -> Output

    Args:
        embed_dim: Dimension size of the hidden embedding
        classes: Number of classification classes in the dataset
        dropout: Dropout value for regularization (Default=0.1)

    Methods:
        forward(inp):
            Applies the sequence of operations mentioned above.
            Input shape: (batch_size, embed_dim)
            Output shape: (batch_size, classes)

    Example:
        >>> head = ClassificationHead(embed_dim=512, classes=10, dropout=0.1)
        >>> logits = head(x)
    '''
    def __init__(self, embed_dim, classes, dropout=0.1):
        super(ClassificationHead, self).__init__()
        self.embed_dim = embed_dim
        self.classes = classes
        self.fc1 = nn.Linear(embed_dim, embed_dim // 2)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(embed_dim // 2, classes)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inp):
        x = self.fc1(inp)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x 
    

class VisionTransformer(nn.Module):
    '''
    Vision Transformer: End-to-end ViT model.
    Applies patch embedding, positional embedding, Transformer layers, and classification head.

    Args:
        inp_channels: Number of channels in the input image
        patch_size: Size of square patch (e.g., 16x16)
        max_len: Max number of tokens (i.e., num_patches + 1 for cls token)
        embed_dim: Embedding dimension
        mlp_dim: MLP hidden dimension
        heads: Number of attention heads
        layers: Number of Transformer blocks
        classes: Number of output classes
        dropout: Dropout probability
    '''
    def __init__(self, inp_channels, patch_size, max_len, heads, classes,
                 layers, embed_dim, mlp_dim, dropout=0.1):
        super(VisionTransformer, self).__init__()

        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Flatten image into patches using a Conv2d layer
        self.patch_embedding = nn.Conv2d(inp_channels, embed_dim,
                                         kernel_size=patch_size,
                                         stride=patch_size)

        # Learnable class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional embedding (learnable)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        self.dropout = nn.Dropout(dropout)

        # Transformer Encoder
        self.transformer = Transformer(embed_dim, layers, heads=heads, dropout=dropout, mlp_dim=mlp_dim)

        # Classification Head (applied to cls token)
        self.classifier = ClassificationHead(embed_dim, classes, dropout)

    def forward(self, inp):
        # inp: (batch_size, channels, height, width)
        batch_size = inp.size(0)

        # Create patch embeddings
        x = self.patch_embedding(inp)  # (batch_size, embed_dim, H', W')
        x = x.flatten(2)  # (batch_size, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (batch_size, num_patches, embed_dim)

        # Prepare class token and concatenate
        cls_tokens = self.cls_token.expand(batch_size, 1, -1)  # (batch_size, 1, embed_dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch_size, seq_len + 1, embed_dim)

        # Add positional embeddings
        x = x + self.pos_embedding[:, :x.size(1), :]
        x = self.dropout(x)

        # Pass through Transformer
        x = self.transformer(x)  # (batch_size, seq_len + 1, embed_dim)

        # Classification output from [CLS] token
        cls_out = x[:, 0]  # (batch_size, embed_dim)
        class_logits = self.classifier(cls_out)  # (batch_size, classes)

        return class_logits, x  # classification result and full transformer output

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# Hyperparameters
EPOCHS = 10
BATCH_SIZE = 64
LR = 3e-4
IMAGE_SIZE = 32
PATCH_SIZE = 4
NUM_CLASSES = 10
EMBED_DIM = 256
MLP_DIM = 512
HEADS = 8
LAYERS = 12
DROPOUT = 0.1
INPUT_CHANNELS = 3
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2
SEQ_LEN = NUM_PATCHES + 1  # for class token

# Data loader
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset  = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model
vit_model = VisionTransformer(
    inp_channels=INPUT_CHANNELS,
    patch_size=PATCH_SIZE,
    max_len=SEQ_LEN,
    embed_dim=EMBED_DIM,
    mlp_dim=MLP_DIM,
    heads=HEADS,
    classes=NUM_CLASSES,
    layers=LAYERS,
    dropout=DROPOUT
).to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(vit_model.parameters(), lr=LR)

# Training loop
for epoch in range(EPOCHS):
    vit_model.train()
    total_loss = 0
    correct = 0
    total = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs, _ = vit_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        loop.set_postfix(loss=loss.item(), acc=100 * correct / total)

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f} | Accuracy: {100 * correct / total:.2f}%")

# Evaluation
vit_model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs, _ = vit_model(images)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

print(f"\n Final Test Accuracy: {100 * correct / total:.2f}%")
