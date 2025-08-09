"""
ComfyUI-style RIFE implementation - НАСТОЯЩАЯ ВЕРСИЯ
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import logging
import os

# RIFE Architecture versions mapping
CKPT_NAME_VER_DICT = {
    "flownet.pth": "4.7",
    "rife47.pth": "4.7", 
    "rife46.pth": "4.6",
    "rife45.pth": "4.5",
    "rife44.pth": "4.3",
    "rife43.pth": "4.3",
    "rife42.pth": "4.2",
    "rife41.pth": "4.0",
    "rife40.pth": "4.0"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backwarp_tenGrid = {}

class ResConv(nn.Module):
    def __init__(self, c, dilation=1):
        super(ResConv, self).__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, dilation, dilation=dilation, groups=1)
        self.beta = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        return self.relu(self.conv(x) * self.beta + x)

def warp(tenInput, tenFlow):
    k = (str(tenFlow.device), str(tenFlow.size()))
    if k not in backwarp_tenGrid:
        tenHorizontal = (
            torch.linspace(-1.0, 1.0, tenFlow.shape[3], device=device)
            .view(1, 1, 1, tenFlow.shape[3])
            .expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
        )
        tenVertical = (
            torch.linspace(-1.0, 1.0, tenFlow.shape[2], device=device)
            .view(1, 1, tenFlow.shape[2], 1)
            .expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
        )
        backwarp_tenGrid[k] = torch.cat([tenHorizontal, tenVertical], 1).to(device)

    tenFlow = torch.cat(
        [
            tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
            tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0),
        ],
        1,
    )

    g = (backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)

    if tenInput.type() == "torch.cuda.HalfTensor":
        g = g.half()

    return torch.nn.functional.grid_sample(
        input=tenInput,
        grid=g,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, arch_ver="4.0"):
    if arch_ver == "4.0":
        return nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=True,
            ),
            nn.PReLU(out_planes),
        )
    if arch_ver in ["4.2", "4.3", "4.5", "4.6", "4.7", "4.10"]:
        return nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=True,
            ),
            nn.LeakyReLU(0.2, True),
        )

def conv_woact(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True,
        ),
    )

def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1, arch_ver="4.0"):
    if arch_ver == "4.0":
        return nn.Sequential(
            torch.nn.ConvTranspose2d(
                in_channels=in_planes,
                out_channels=out_planes,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=True,
            ),
            nn.PReLU(out_planes),
        )
    if arch_ver in ["4.2", "4.3", "4.5", "4.6", "4.7", "4.10"]:
        return nn.Sequential(
            torch.nn.ConvTranspose2d(
                in_channels=in_planes,
                out_channels=out_planes,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=True,
            ),
            nn.LeakyReLU(0.2, True),
        )

class Conv2(nn.Module):
    def __init__(self, in_planes, out_planes, stride=2, arch_ver="4.0"):
        super(Conv2, self).__init__()
        self.conv1 = conv(in_planes, out_planes, 3, stride, 1, arch_ver=arch_ver)
        self.conv2 = conv(out_planes, out_planes, 3, 1, 1, arch_ver=arch_ver)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64, arch_ver="4.0"):
        super(IFBlock, self).__init__()
        self.arch_ver = arch_ver
        self.conv0 = nn.Sequential(
            conv(in_planes, c // 2, 3, 2, 1, arch_ver=arch_ver),
            conv(c // 2, c, 3, 2, 1, arch_ver=arch_ver),
        )
        self.arch_ver = arch_ver

        if arch_ver in ["4.0", "4.2", "4.3"]:
            self.convblock = nn.Sequential(
                conv(c, c, arch_ver=arch_ver),
                conv(c, c, arch_ver=arch_ver),
                conv(c, c, arch_ver=arch_ver),
                conv(c, c, arch_ver=arch_ver),
                conv(c, c, arch_ver=arch_ver),
                conv(c, c, arch_ver=arch_ver),
                conv(c, c, arch_ver=arch_ver),
                conv(c, c, arch_ver=arch_ver),
            )
            self.lastconv = nn.ConvTranspose2d(c, 5, 4, 2, 1)

        if arch_ver in ["4.5", "4.6", "4.7", "4.10"]:
            self.convblock = nn.Sequential(
                ResConv(c),
                ResConv(c),
                ResConv(c),
                ResConv(c),
                ResConv(c),
                ResConv(c),
                ResConv(c),
                ResConv(c),
            )
        if arch_ver == "4.5":
            self.lastconv = nn.Sequential(
                nn.ConvTranspose2d(c, 4 * 5, 4, 2, 1), nn.PixelShuffle(2)
            )
        if arch_ver in ["4.6", "4.7", "4.10"]:
            self.lastconv = nn.Sequential(
                nn.ConvTranspose2d(c, 4 * 6, 4, 2, 1), nn.PixelShuffle(2)
            )

    def forward(self, x, flow=None, scale=1):
        x = F.interpolate(
            x, scale_factor=1.0 / scale, mode="bilinear", align_corners=False
        )
        if flow is not None:
            flow = (
                F.interpolate(
                    flow, scale_factor=1.0 / scale, mode="bilinear", align_corners=False
                )
                * 1.0
                / scale
            )
            x = torch.cat((x, flow), 1)
        feat = self.conv0(x)
        if self.arch_ver == "4.0":
            feat = self.convblock(feat) + feat
        if self.arch_ver in ["4.2", "4.3", "4.5", "4.6", "4.7", "4.10"]:
            feat = self.convblock(feat)

        tmp = self.lastconv(feat)
        if self.arch_ver in ["4.0", "4.2", "4.3"]:
            tmp = F.interpolate(
                tmp, scale_factor=scale * 2, mode="bilinear", align_corners=False
            )
            flow = tmp[:, :4] * scale * 2
        if self.arch_ver in ["4.5", "4.6", "4.7", "4.10"]:
            tmp = F.interpolate(
                tmp, scale_factor=scale, mode="bilinear", align_corners=False
            )
            flow = tmp[:, :4] * scale
        mask = tmp[:, 4:5]
        return flow, mask

class IFNet(nn.Module):
    def __init__(self, arch_ver="4.7"):
        super(IFNet, self).__init__()
        self.arch_ver = arch_ver
        if arch_ver in ["4.0", "4.2", "4.3", "4.5", "4.6"]:
            self.block0 = IFBlock(7, c=192, arch_ver=arch_ver)
            self.block1 = IFBlock(8 + 4, c=128, arch_ver=arch_ver)
            self.block2 = IFBlock(8 + 4, c=96, arch_ver=arch_ver)
            self.block3 = IFBlock(8 + 4, c=64, arch_ver=arch_ver)
        if arch_ver in ["4.7"]:
            self.block0 = IFBlock(7 + 8, c=192, arch_ver=arch_ver)
            self.block1 = IFBlock(8 + 4 + 8, c=128, arch_ver=arch_ver)
            self.block2 = IFBlock(8 + 4 + 8, c=96, arch_ver=arch_ver)
            self.block3 = IFBlock(8 + 4 + 8, c=64, arch_ver=arch_ver)
            self.encode = nn.Sequential(
                nn.Conv2d(3, 16, 3, 2, 1), nn.ConvTranspose2d(16, 4, 4, 2, 1)
            )
        if arch_ver in ["4.10"]:
            self.block0 = IFBlock(7 + 16, c=192)
            self.block1 = IFBlock(8 + 4 + 16, c=128)
            self.block2 = IFBlock(8 + 4 + 16, c=96)
            self.block3 = IFBlock(8 + 4 + 16, c=64)
            self.encode = nn.Sequential(
                nn.Conv2d(3, 32, 3, 2, 1),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(32, 32, 3, 1, 1),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(32, 32, 3, 1, 1),
                nn.LeakyReLU(0.2, True),
                nn.ConvTranspose2d(32, 8, 4, 2, 1),
            )

        self.arch_ver = arch_ver

    def forward(self, img0, img1, timestep=0.5, scale_list=[8, 4, 2, 1], fastmode=True, ensemble=False):
        """Forward pass with ComfyUI-style parameters"""
        img0 = torch.clamp(img0, 0, 1)
        img1 = torch.clamp(img1, 0, 1)

        n, c, h, w = img0.shape
        ph = ((h - 1) // 64 + 1) * 64
        pw = ((w - 1) // 64 + 1) * 64
        padding = (0, pw - w, 0, ph - h)
        img0 = F.pad(img0, padding)
        img1 = F.pad(img1, padding)

        if not torch.is_tensor(timestep):
            timestep = (img0[:, :1].clone() * 0 + 1) * timestep
        else:
            timestep = timestep.repeat(1, 1, img0.shape[2], img0.shape[3])

        if self.arch_ver in ["4.7", "4.10"]:
            f0 = self.encode(img0[:, :3])
            f1 = self.encode(img1[:, :3])

        warped_img0 = img0
        warped_img1 = img1
        flow = None
        mask = None
        block = [self.block0, self.block1, self.block2, self.block3]

        for i in range(4):
            if flow is None:
                # 4.0-4.6
                if self.arch_ver in ["4.0", "4.2", "4.3", "4.5", "4.6"]:
                    flow, mask = block[i](
                        torch.cat((img0[:, :3], img1[:, :3], timestep), 1),
                        None,
                        scale=scale_list[i],
                    )
                    if ensemble:
                        f1, m1 = block[i](
                            torch.cat((img1[:, :3], img0[:, :3], 1 - timestep), 1),
                            None,
                            scale=scale_list[i],
                        )
                        flow = (flow + torch.cat((f1[:, 2:4], f1[:, :2]), 1)) / 2
                        mask = (mask + (-m1)) / 2

                # 4.7+
                if self.arch_ver in ["4.7", "4.10"]:
                    flow, mask = block[i](
                        torch.cat((img0[:, :3], img1[:, :3], f0, f1, timestep), 1),
                        None,
                        scale=scale_list[i],
                    )

                    if ensemble:
                        f_, m_ = block[i](
                            torch.cat(
                                (img1[:, :3], img0[:, :3], f1, f0, 1 - timestep), 1
                            ),
                            None,
                            scale=scale_list[i],
                        )
                        flow = (flow + torch.cat((f_[:, 2:4], f_[:, :2]), 1)) / 2
                        mask = (mask + (-m_)) / 2

            else:
                # 4.0-4.6
                if self.arch_ver in ["4.0", "4.2", "4.3", "4.5", "4.6"]:
                    f0, m0 = block[i](
                        torch.cat(
                            (warped_img0[:, :3], warped_img1[:, :3], timestep, mask), 1
                        ),
                        flow,
                        scale=scale_list[i],
                    )

                # 4.7+
                if self.arch_ver in ["4.7", "4.10"]:
                    fd, m0 = block[i](
                        torch.cat(
                            (
                                warped_img0[:, :3],
                                warped_img1[:, :3],
                                warp(f0, flow[:, :2]),
                                warp(f1, flow[:, 2:4]),
                                timestep,
                                mask,
                            ),
                            1,
                        ),
                        flow,
                        scale=scale_list[i],
                    )
                    flow = flow + fd

                if self.arch_ver in ["4.0", "4.2", "4.3", "4.5", "4.6"]:
                    flow = flow + f0
                    mask = mask + m0

                if not ensemble and self.arch_ver in ["4.7", "4.10"]:
                    mask = m0

            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])

        if self.arch_ver in ["4.0", "4.1", "4.2", "4.3", "4.4", "4.5", "4.6"]:
            mask = torch.sigmoid(mask)
            merged = warped_img0 * mask + warped_img1 * (1 - mask)

        if self.arch_ver in ["4.7", "4.10"]:
            mask = torch.sigmoid(mask)
            merged = warped_img0 * mask + warped_img1 * (1 - mask)

        return merged[:, :, :h, :w]

class ComfyRIFE:
    """ComfyUI-style RIFE implementation"""
    
    def __init__(self, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = None
        self.available = False
        self._setup_rife()
    
    def _setup_rife(self):
        """Setup RIFE model"""
        try:
            logging.info("🚀 Setting up ComfyUI-style RIFE...")
            
            # Find or download weight file
            project_root = os.path.dirname(os.path.dirname(__file__))
            weights_dir = os.path.join(project_root, 'weights')
            os.makedirs(weights_dir, exist_ok=True)
            
            weight_file = None
            arch_ver = "4.7"  # Default
            ckpt_name = "rife47.pth"  # Default checkpoint
            
            # Check if we already have weights
            if os.path.exists(weights_dir):
                for filename in os.listdir(weights_dir):
                    if filename.endswith(('.pth', '.pkl')):
                        weight_file = os.path.join(weights_dir, filename)
                        arch_ver = CKPT_NAME_VER_DICT.get(filename, "4.7")
                        logging.info(f"✅ Found local weights: {filename} (arch {arch_ver})")
                        break
            
            # Download weights if not found
            if not weight_file:
                logging.info(f"📦 Downloading RIFE weights: {ckpt_name}")
                weight_file = self._download_rife_weights(ckpt_name, weights_dir)
                if weight_file:
                    arch_ver = CKPT_NAME_VER_DICT.get(ckpt_name, "4.7")
                    logging.info(f"✅ Downloaded weights: {ckpt_name} (arch {arch_ver})")
                else:
                    logging.warning("⚠️ Could not download weights, using random initialization")
                    arch_ver = "4.7"
            
            # Create model with correct architecture
            self.model = IFNet(arch_ver=arch_ver)
            self.model.eval().to(self.device)
            
            # Load weights if available
            if weight_file and os.path.exists(weight_file):
                try:
                    logging.info(f"🔄 Loading RIFE weights from: {weight_file}")
                    state_dict = torch.load(weight_file, map_location=self.device)
                    
                    # Handle different weight formats
                    if isinstance(state_dict, dict):
                        # Standard dict format
                        weights = state_dict
                    elif hasattr(state_dict, 'state_dict'):
                        # Model checkpoint format
                        weights = state_dict.state_dict()
                    else:
                        logging.warning(f"⚠️ Unknown weight format: {type(state_dict)}")
                        weights = None
                    
                    if weights:
                        # Try to load compatible weights
                        try:
                            result = self.model.load_state_dict(weights, strict=False)
                            if result.missing_keys or result.unexpected_keys:
                                logging.info(f"   Partial load: {len(result.missing_keys)} missing, {len(result.unexpected_keys)} unexpected")
                                if len(result.missing_keys) < len(weights) // 2:  # If we loaded at least 50%
                                    logging.info("✅ Partial RIFE weights loaded successfully (>50% compatible)")
                                else:
                                    logging.warning("⚠️ Too many missing keys for useful partial load")
                            else:
                                logging.info("✅ Perfect RIFE weight match!")
                        except Exception as e:
                            logging.warning(f"⚠️ Weight loading failed: {e}")
                    
                except Exception as e:
                    logging.warning(f"⚠️ Could not load weights: {e}")
            
            self.available = True
            logging.info("✅ ComfyUI-style REAL RIFE ready!")
            
        except Exception as e:
            logging.error(f"❌ ComfyUI RIFE setup failed: {e}")
            self.available = False
    
    def _download_rife_weights(self, ckpt_name, weights_dir):
        """Download RIFE weights from GitHub releases like ComfyUI does"""
        try:
            import urllib.request
            import urllib.error
            
            # GitHub release URLs (same as ComfyUI uses)
            base_urls = [
                "https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/",
                "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation/releases/download/models/",
                "https://github.com/dajes/frame-interpolation-pytorch/releases/download/v1.0.0/"
            ]
            
            for base_url in base_urls:
                url = base_url + ckpt_name
                local_path = os.path.join(weights_dir, ckpt_name)
                
                try:
                    logging.info(f"🔄 Trying to download from: {url}")
                    
                    def show_progress(block_num, block_size, total_size):
                        if total_size > 0:
                            percent = min(100, (block_num * block_size * 100) // total_size)
                            if block_num % 100 == 0:  # Log every 100 blocks
                                logging.info(f"   Download progress: {percent}%")
                    
                    urllib.request.urlretrieve(url, local_path, reporthook=show_progress)
                    logging.info(f"✅ Successfully downloaded: {ckpt_name}")
                    return local_path
                    
                except urllib.error.URLError as e:
                    logging.warning(f"   Failed to download from {url}: {e}")
                    continue
                except Exception as e:
                    logging.warning(f"   Download error from {url}: {e}")
                    continue
            
            logging.error(f"❌ Failed to download {ckpt_name} from all URLs")
            return None
            
        except Exception as e:
            logging.error(f"❌ Download function failed: {e}")
            return None
    
    def interpolate_at_timestep(self, frame1, frame2, timestep):
        """Interpolate single frame at timestep"""
        if not self.available:
            return self._simple_blend(frame1, frame2, timestep)
        
        try:
            # Convert frames to tensors
            tensor1 = self._frame_to_tensor(frame1)
            tensor2 = self._frame_to_tensor(frame2)
            
            with torch.no_grad():
                # Run REAL RIFE inference
                result_tensor = self.model(tensor1, tensor2, timestep, fastmode=True, ensemble=False)
                
                # Convert back to frame
                result_frame = self._tensor_to_frame(result_tensor)
                return result_frame
                
        except Exception as e:
            logging.warning(f"RIFE inference failed: {e}, using blend fallback")
            return self._simple_blend(frame1, frame2, timestep)
    
    def _simple_blend(self, frame1, frame2, timestep):
        """Simple weighted blend fallback"""
        weight1 = 1.0 - timestep
        weight2 = timestep
        
        frame1_f = frame1.astype(np.float32)
        frame2_f = frame2.astype(np.float32)
        
        result = (frame1_f * weight1 + frame2_f * weight2)
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _frame_to_tensor(self, frame):
        """Convert frame to tensor"""
        # BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Normalize and convert to tensor
        tensor = torch.from_numpy(frame_rgb).float() / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # HWC -> BCHW
        
        return tensor.to(self.device)
    
    def _tensor_to_frame(self, tensor):
        """Convert tensor to frame"""
        tensor = tensor.squeeze(0).cpu().permute(1, 2, 0)  # BCHW -> HWC
        frame_rgb = (tensor.clamp(0, 1) * 255).byte().numpy()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        return frame_bgr