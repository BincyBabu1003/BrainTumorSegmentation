import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# ----------------------------
# Double Convolution Block
# ----------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)


# ----------------------------
# U-Net Model
# ----------------------------
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.down1 = DoubleConv(1, 64)
        self.down2 = DoubleConv(64, 128)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(128, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv1 = DoubleConv(128, 64)

        self.final = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))

        b = self.bottleneck(self.pool(d2))

        u2 = self.up2(b)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.conv2(u2)

        u1 = self.up1(u2)
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.conv1(u1)

        return torch.sigmoid(self.final(u1))


# ----------------------------
# Load Model
# ----------------------------
model = UNet()

model.load_state_dict(
    torch.load(
        "model (1).pth",
        map_location=torch.device("cpu")
    )
)



model.eval()


# ----------------------------
# Streamlit UI
# ----------------------------
st.title("Brain Tumor Segmentation using U-Net")

uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert("L")
    image = image.resize((256, 256))

    img_array = np.array(image) / 255.0

    st.subheader("Uploaded MRI Image")
    st.image(img_array, caption="Original MRI", use_container_width=True)

    # Convert to tensor
    input_tensor = torch.from_numpy(img_array).float().unsqueeze(0).unsqueeze(0)

    # Predict
    with torch.no_grad():
        pred_mask = model(input_tensor)

    pred_mask = pred_mask.squeeze().numpy()
    pred_mask = (pred_mask > 0.5).astype(float)

    st.subheader("Predicted Tumor Mask")
    st.image(pred_mask, caption="Predicted Mask", use_container_width=True)

    # ⭐ NEW: Large standalone tumor mask
    st.subheader("Tumor Mask (Large View)")
    st.image(pred_mask, caption="Large Tumor Mask", use_container_width=True)

    # Overlay Visualization
    st.subheader("MRI + Predicted Mask Overlay")

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    ax[0].imshow(image, cmap="gray")
    ax[0].set_title("Input Image")
    ax[0].axis("off")

    ax[1].imshow(img_array, cmap="gray")
    ax[1].set_title("Gray MRI Image")
    ax[1].axis("off")

    ax[2].imshow(img_array, cmap="gray")
    ax[2].imshow(pred_mask, cmap="Reds", alpha=0.5)
    ax[2].set_title("Segmented Tumor")
    ax[2].axis("off")

    st.pyplot(fig)
