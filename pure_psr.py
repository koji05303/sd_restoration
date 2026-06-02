import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import urllib.request
import gc


DEFAULT_CUDA_DEVICE = "cuda:0"  

"""
TODO:
- [x] 手刻 RRDBNet 架構
- [x] 對齊 Real-ESRGAN_x4plus 的官方權重矩陣
- [x] 實作分塊推理 (tile-based
- [x] 增加邊界 padding 避免偽影 and 無縫拼接
- [x] 支援 FP16 加速 (如果 GPU 支援的話)
- [x] 增加 USM 銳化選項
- [x] 測試大圖無 OOM, 且輸出品質與官方一致

- [x] 物件封裝成 PureSREngine 類別，提供 enhance() 方法接受 OpenCV BGR 圖片並返回放大後的圖片。
- [x] 建立 Gradio Based UI，允許用戶上傳圖片並選擇是否啟用細節增強，然後顯示放大後的結果。需要有 before/after 對比視窗。
- [x] 網頁端須即時顯示 gpu 狀態，例如 vram 使用量，還有當前處理進度條和預估處理時間。
- [x] 能作多好看就多好看，UI/UX 設計要簡潔直觀，盡量模仿專業圖像處理軟體的風格，讓人一看就懂怎麼用。
"""

# ==========================================
# 手刻神經網路架構 (RRDBNet)
# 對齊 Real-ESRGAN_x4plus 的官方權重矩陣
# ==========================================

class DenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x # 殘差縮放

class RRDB(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.rdb1 = DenseBlock(num_feat, num_grow_ch)
        self.rdb2 = DenseBlock(num_feat, num_grow_ch)
        self.rdb3 = DenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x

class CustomRRDBNet(nn.Module):
    def __init__(self):
        super().__init__()
        num_feat, num_grow_ch, num_block = 64, 32, 23
        self.conv_first = nn.Conv2d(3, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # 兩次雙倍上採樣, total x4
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, 3, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        # 上採樣
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out



### PSR Engine, code from scratch
class PureSREngine:
    def __init__(self, device=DEFAULT_CUDA_DEVICE, model_url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"):
        self.device = torch.device(device)
        self.scale = 4
        # fp16, 省省省屁眼汁
        self.half = self.device.type == "cuda" and torch.cuda.is_available()
                
        # 由於開啟了 half 精度，tile_size可以高一咪咪, 建議是別開太大，我先抓256 避免屁眼開花
        self.tile_size = 256
        # 增加 padding 避免邊界偽影 and 無縫拼接
        self.tile_pad = 16   
        
        print(f"Pure SR 引擎 ==> 綁定設備: {device} | 精度: {'FP16' if self.half else 'FP32'}")
        self.model = CustomRRDBNet()
        self._load_weights(model_url) 

        self.model.eval()
        self.model.requires_grad_(False)

        if self.half:
            self.model = self.model.half()

        self.model = self.model.to(self.device)

    def _load_weights(self, url):
        
        weight_path = "RealESRGAN_x4plus.pth"

        if not os.path.exists(weight_path):
            print(f"Pulling Original weights from {url}...")
            urllib.request.urlretrieve(url, weight_path)
            
        print("Recreating the model arc, Matrix loading...")
        
        loadnet = torch.load(weight_path, map_location="cpu")
        
        keyname = "params_ema" if "params_ema" in loadnet else "params"
        self.model.load_state_dict(loadnet[keyname], strict=True)

        del loadnet
        gc.collect()
        
        print("Node Inject complete.")

    @torch.inference_mode()
    def enhance(self, img_bgr, enhance_detail=True, progress_callback=None):
        """輸入 OpenCV BGR 圖片，吐出 4 倍無損放大圖片"""
        #BGR -> RGB -> Float Tensor [1, 3, H, W]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
        
        if self.half:
            img_tensor = img_tensor.half()

        _, channel, height, width = img_tensor.shape
        output_height = height * self.scale
        output_width = width * self.scale
        
        ## Canvas stays in CPU memory as uint8 to reduce host RAM pressure on large outputs.
        output_numpy_rgb = np.empty((output_height, output_width, channel), dtype=np.uint8)
        total_tiles = ((height + self.tile_size - 1) // self.tile_size) * (
            (width + self.tile_size - 1) // self.tile_size
        )
        tile_idx = 0
        cuda_clean_interval = 8
        gc_clean_interval = 32

        # sliding window with tile and padding to process large images without OOM,
        for y in range(0, height, self.tile_size):
            for x in range(0, width, self.tile_size):
                tile_idx += 1
                # 取得當前區塊的邊界
                y_end = min(y + self.tile_size, height)
                x_end = min(x + self.tile_size, width)
                
                # 計算包含 padding 的擴展邊界 (處理邊緣不越界)
                y_pad_start = max(y - self.tile_pad, 0)
                y_pad_end = min(y_end + self.tile_pad, height)
                x_pad_start = max(x - self.tile_pad, 0)
                x_pad_end = min(x_end + self.tile_pad, width)
                
                # 切出帶有 padding 的小張量送去推理, stays on CPU
                input_tile = img_tensor[:, :, y_pad_start:y_pad_end, x_pad_start:x_pad_end]      

                ### send tile to GPU for processing
                input_tile = input_tile.to(self.device)
                if self.half:
                    input_tile = input_tile.half()

              
                with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.half):  # 丟進自建的 RRDBNet 進行前向爆破 (搭配 AMP 或純 FP16)
                    output_tile = self.model(input_tile)
                    
                # 算完了是吧 那把剛padding的部份切掉 只留下原圖區域
                y_out_start = y * self.scale
                y_out_end = y_end * self.scale
                x_out_start = x * self.scale
                x_out_end = x_end * self.scale
                
                y_crop_start = (y - y_pad_start) * self.scale
                y_crop_end = y_crop_start + (y_end - y) * self.scale
                x_crop_start = (x - x_pad_start) * self.scale
                x_crop_end = x_crop_start + (x_end - x) * self.scale
                
                # Crop on GPU first, then move only the valid area back to CPU.
                output_crop = output_tile[:, :, y_crop_start:y_crop_end, x_crop_start:x_crop_end]
                output_crop = output_crop.detach().float().cpu()
                output_crop = output_crop.squeeze(0).permute(1, 2, 0).clamp_(0, 1)
                output_crop_np = (output_crop.numpy() * 255.0).round().astype(np.uint8)

                output_numpy_rgb[y_out_start:y_out_end, x_out_start:x_out_end, :] = output_crop_np

                del input_tile, output_tile, output_crop, output_crop_np
                if self.device.type == "cuda" and tile_idx % cuda_clean_interval == 0:
                    torch.cuda.empty_cache()
                if tile_idx % gc_clean_interval == 0:
                    gc.collect()
                if progress_callback:
                    progress_callback(tile_idx, total_tiles)

        # Numpy RGB -> BGR
        output_bgr = cv2.cvtColor(output_numpy_rgb, cv2.COLOR_RGB2BGR)
        
        # Unsharp Masking (USM)
        if enhance_detail:
            # Gaussaingn Blur, sigma=2.0, kernel size auto
            blur = cv2.GaussianBlur(output_bgr, (0, 0), 2.0)
            # add weighted: output + (output - blur) * 1.5, alpha=1.5, beta=-0.5
            output_bgr = cv2.addWeighted(output_bgr, 1.5, blur, -0.5, 0)
        
        return output_bgr

if __name__ == '__main__':
    
    engine = PureSREngine(device=DEFAULT_CUDA_DEVICE)
    
    input_img = cv2.imread("input/fake.jpg")
    if input_img is not None:
        print("Starting Pure SR Matrix upscaling...")
        import time
        t0 = time.time()
        
        result = engine.enhance(input_img)
        
        # result = cv2.resize(result, (result.shape[1]//2, result.shape[0]//2), interpolation=cv2.INTER_LANCZOS4)
        
        cv2.imwrite("output/fake_pure_sr.jpg", result)
        print(f"Perfectly generated! Time taken: {time.time() - t0:.3f} seconds")
    else:
        print("Damn, couldn't read the image! Check your path!")
