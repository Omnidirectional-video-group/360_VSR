import cv2
import numpy as np
import math
import os
import pandas as pd
from statistics import mean
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from tqdm import tqdm
import concurrent.futures
import sys
import argparse

import time

def calculate_psnr_ws(img, img2, crop_border, input_order='HWC', **kwargs):
    """
    Calculate weighted PSNR between two images.
    img, img2: Input images for comparison.
    crop_border: Border width to crop from images before calculation.
    input_order: Format of input images ('HWC' or 'CHW').
    """
    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)
    img_w = compute_map_ws(img)

    mse = np.mean(np.multiply((img - img2)**2, img_w))/np.mean(img_w)
    if mse == 0:
        return float('inf')
    return 10. * np.log10(255. * 255. / mse)

def calculate_ssim_ws(img, img2, crop_border, input_order='HWC', **kwargs):
    """
    Calculate weighted SSIM between two images.
    img, img2: Input images for comparison.
    crop_border: Border width to crop from images before calculation.
    input_order: Format of input images ('HWC' or 'CHW').
    """
    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')
    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)

    ssims = []
    for i in range(img.shape[2]):
        ssims.append(_ws_ssim(img[..., i], img2[..., i]))
    return np.array(ssims).mean()

def genERP(j,N):
    """
    Generate weight for ERP (Equirectangular Projection) given pixel position.
    j: Pixel position.
    N: Total number of pixels in the dimension.
    """
    val = math.pi/N
    w = math.cos((j - (N/2) + 0.5) * val)
    return w


def compute_map_ws(img):
    """
    Compute the weighting map for weighted metrics calculation.
    img: Input image to calculate the weight map.
    """
    height = img.shape[0]
    y_indices = np.arange(height) - (height / 2) + 0.5
    w = np.cos(np.pi * y_indices / height)
    return np.tile(w[:, np.newaxis, np.newaxis], (1, img.shape[1], img.shape[2]))


def _ws_ssim(img, img2):
    """
    Internal function to calculate weighted SSIM for a single channel.
    img, img2: Single channel images for SSIM calculation.
    """
    c1 = (0.01 * 255)**2
    c2 = (0.03 * 255)**2
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img, -1, window)[5:-5, 5:-5]  # valid mode for window size 11
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))

    equ = np.zeros((ssim_map.shape[0], ssim_map.shape[1]))

    for i in range(0,equ.shape[0]):
        for j in range(0,equ.shape[1]):
                equ[i, j] = genERP(i,equ.shape[0])

    return np.multiply(ssim_map, equ).mean()/equ.mean()



def reorder_image(img, input_order='HWC'):
    """
    Reorder the image from HWC to CHW format or vice versa.
    img: Image to be reordered.
    input_order: Current format of the image ('HWC' or 'CHW').
    """
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f"Wrong input_order {input_order}. Supported input_orders are 'HWC' and 'CHW'")
    if len(img.shape) == 2:
        img = img[..., None]
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)
    return img





def mean(values):
    return sum(values) / len(values)

def process_frame(HR_image, LW_image, original_width, original_height, weight_map, frame_num, video_num, LW_video_path):
    """
    Process a single frame to calculate quality metrics.
    HR_image, LW_image: High resolution and upscaled low resolution images of the frame.
    original_width, original_height: Dimensions of the original image.
    weight_map: Weighting map for calculation.
    frame_num: Frame number.
    video_num: Video number.
    LW_video_path: Path of the low-resolution video.
    """
    results = {}
    
    results[f'frame_{frame_num}_wspsnr'] = calculate_psnr_ws(HR_image, LW_image, crop_border=0, weight_map=weight_map)
    results[f'frame_{frame_num}_wsssim'] = calculate_ssim_ws(HR_image, LW_image, crop_border=0, weight_map=weight_map)
    
    return results
    
def process_video(video_num, HR_dir, base_LW_dir):
    """
    Process a video to calculate quality metrics for each frame.
    video_num: Number of the video to be processed.
    """
    print(f"Processing Video {video_num}")
    
    video_name = video_num
    HR_video_path = os.path.join(HR_dir, video_name)

    sample_frame = cv2.imread(f'./{HR_dir}/{video_name}/001.png')
    weight_map = compute_map_ws(sample_frame)

    # Create a dictionary to store results for each bitrate
    bitrate_results = {
        '1': [],
        '2': [],
        '3': [],
        '4': []
    }
    LW_dirs = get_subdirectories(base_LW_dir)
    # Paths for different bitrate videos
    LW_videos_paths = {
        '1': os.path.join(LW_dirs[0], video_name),
        '2': os.path.join(LW_dirs[1], video_name),
        '3': os.path.join(LW_dirs[2], video_name),
        '4': os.path.join(LW_dirs[3], video_name)
    }

    # Process frames for each bitrate
    for bitrate, LW_video_path in LW_videos_paths.items():
        for frame_num in range(1, 100, 10):
            try:
                frame_name = f"{frame_num:03d}.png"

                HR_image = cv2.imread(os.path.join(HR_video_path, frame_name))
                LW_image = cv2.imread(os.path.join(LW_video_path, frame_name))


                if HR_image.shape != weight_map.shape:
                    print('Weight map has different shape of HR image!')
                    sys.exit()

                if HR_image is None or LW_image is None:
                    raise ValueError(f"Error reading frame {frame_num}")

                original_height, original_width = LW_image.shape[:2]
                frame_results = process_frame(HR_image, LW_image, original_width, original_height, weight_map, frame_num, video_num, LW_video_path)
                bitrate_results[bitrate].append(frame_results)
            except Exception as e:
                print(f"Error processing frame {frame_num} in video {video_num}: {e}")
                continue


    # Calculate average results for each bitrate and metric
    metrics = [ 'wspsnr', 'wsssim']
    print(bitrate_results)
    print(frame)
    avg_results = {bitrate: {metric: mean([frame[f'frame_{i}_{metric}'] 
                                            for i in range(1, 101, 10) 
                                            for frame in frames 
                                            if f'frame_{i}_{metric}' in frame]) 
                             for metric in metrics} 
                   for bitrate, frames in bitrate_results.items()}

    # Transform avg_results to structured_results format
    structured_results = {
        'Video Name': video_name,
        'Bitrate': [],
        'WS-PSNR': [],
        'WS-SSIM': []
    }
    
    for bitrate, metrics in avg_results.items():
        structured_results['Bitrate'].append(bitrate)
        structured_results['WS-PSNR'].append(metrics['wspsnr'])
        structured_results['WS-SSIM'].append(metrics['wsssim'])

    return structured_results




def save_results_to_excel(results, filename):
    """Saves results in an excel file"""
    df = pd.DataFrame(results)
    df.to_excel(filename, index=False, engine='openpyxl')

def get_subdirectories(base_dir):
    """Returns a list of subdirectory names in the given base directory"""
    return [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

def main(HR_dir, base_LW_dir, num_processes):
    """Main function"""

    video_numbers = os.listdir(HR_dir)

    all_video_results = []  

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [executor.submit(process_video, video_num, HR_dir, base_LW_dir) for video_num in video_numbers]

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(video_numbers), desc="Processing Videos"):
            video_results = future.result()
            all_video_results.append(video_results)  

    # Combine all results into a single DataFrame
    combined_results = pd.concat([pd.DataFrame(video_result) for video_result in all_video_results], ignore_index=True)
    avg_ws_psnr = combined_results['WS-PSNR'].mean()
    avg_ws_ssim = combined_results['WS-SSIM'].mean()

    print(f"Average WS-PSNR across all videos and bitrates: {avg_ws_psnr}")
    print(f"Average WS-SSIM across all videos and bitrates: {avg_ws_ssim}")

    #Save the combined results to a single Excel file
    #excel_filename = 'video_X2_quality_analysis_test.xlsx'
    #combined_results.to_excel(excel_filename, index=False, engine='openpyxl')
    print(f"Computed metrics compared to {base_LW_dir}")





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process videos')
    parser.add_argument('--HR_dir', type=str, required=True, help='High Resolution Directory')
    parser.add_argument('--LW_dir', type=str, required=True, help='Base directory for Low Resolution subdirectories')
    parser.add_argument('--num_processes', type=int, default=20, help='Number of processes to use (default: all available CPUs)')

    args = parser.parse_args()
    
    main(args.HR_dir, args.LW_dir, args.num_processes)