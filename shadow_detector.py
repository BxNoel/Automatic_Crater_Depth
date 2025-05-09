import cv2
import numpy as np
import os
import matplotlib.pyplot as plt # Still useful for optional display of one image

def process_image_for_shadows(image_path, 
                             method="adaptive",  # 'otsu', 'adaptive', or 'canny'
                             enhance_contrast=True,
                             dark_shadow_percentage=10,  # For darkness-based method
                             adaptive_block_size=51,  # For adaptive method
                             adaptive_constant=5,  # For adaptive method
                             apply_morphology=True,
                             morph_operation="close",  # 'open', 'close', 'dilate', 'erode'
                             morph_kernel_size=5,
                             morph_iterations=1):
    """
    Advanced shadow detection in crater images using various methods.
    
    Args:
        image_path (str): The path to the image file.
        method (str): Detection method ('otsu', 'adaptive', or 'canny')
        enhance_contrast (bool): Whether to enhance image contrast before thresholding
        dark_shadow_percentage (int): Percentage of darkest pixels to use as shadows (1-20 recommended)
        adaptive_block_size (int): Block size for adaptive thresholding (must be odd)
        adaptive_constant (int): Constant subtracted from mean in adaptive thresholding
        apply_morphology (bool): Whether to apply morphological operations to refine the mask
        morph_operation (str): Type of morphology operation ('open', 'close', 'dilate', 'erode')
        morph_kernel_size (int): Size of morphology kernel (must be odd)
        morph_iterations (int): Number of times to apply morphology operation

    Returns:
        A tuple (original_image_rgb, gray_image, shadow_mask, threshold_value, adaptive_threshold_value)
        or (None, None, None, None, None) if loading fails.
        The shadow_mask will have shadows as white (255) and background as black (0).
    """
    original_image_bgr = cv2.imread(image_path)

    if original_image_bgr is None:
        print(f"Warning: Could not read image at {image_path}")
        return None, None, None, None, None

    # Convert to grayscale
    gray_image = cv2.cvtColor(original_image_bgr, cv2.COLOR_BGR2GRAY)
    
    # Store the original for comparison
    gray_original = gray_image.copy()
    
    # Enhance contrast if requested
    if enhance_contrast:
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray_image = clahe.apply(gray_image)
    
    # Apply Gaussian blur to reduce noise
    gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    # Get Otsu threshold (mostly for reference/comparison)
    otsu_thresh_val, _ = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Method 1: Basic Otsu thresholding
    if method == "otsu":
        _, shadow_mask = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        adaptive_thresh_val = otsu_thresh_val
        
    # Method 2: Adaptive thresholding (usually better for uneven lighting)
    elif method == "adaptive":
        # Ensure block size is odd
        if adaptive_block_size % 2 == 0:
            adaptive_block_size += 1
            
        shadow_mask = cv2.adaptiveThreshold(
            gray_image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            adaptive_block_size,
            adaptive_constant
        )
        adaptive_thresh_val = adaptive_constant  # Not the actual threshold, just for reference
    
    # Method 3: Percentage-based dark pixel detection
    elif method == "darkness":
        # Sort pixels by intensity and take the darkest n%
        flat_gray = gray_image.flatten()
        threshold_index = int(len(flat_gray) * dark_shadow_percentage / 100)
        sorted_pixels = np.sort(flat_gray)
        if threshold_index >= len(sorted_pixels):
            threshold_index = len(sorted_pixels) - 1
        adaptive_thresh_val = sorted_pixels[threshold_index]
        
        _, shadow_mask = cv2.threshold(gray_image, adaptive_thresh_val, 255, cv2.THRESH_BINARY_INV)
    
    # Method 4: Edge detection (Canny) - useful for detecting shadow boundaries
    elif method == "canny":
        # Auto-calculate thresholds based on median of the image
        median_val = np.median(gray_image)
        lower = int(max(0, (1.0 - 0.33) * median_val))
        upper = int(min(255, (1.0 + 0.33) * median_val))
        
        edges = cv2.Canny(gray_image, lower, upper)
        # Dilate edges to connect components
        kernel = np.ones((3,3), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours and fill them
        contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        shadow_mask = np.zeros_like(gray_image)
        for contour in contours:
            # Filter by contour area if needed
            area = cv2.contourArea(contour)
            if area > 100:  # Minimum area threshold
                cv2.drawContours(shadow_mask, [contour], 0, 255, -1)  # -1 means fill
        
        adaptive_thresh_val = 0  # Not applicable for this method
    
    else:  # Default to adaptive if invalid method
        print(f"Warning: Unknown method '{method}'. Using adaptive thresholding.")
        if adaptive_block_size % 2 == 0:
            adaptive_block_size += 1
        shadow_mask = cv2.adaptiveThreshold(
            gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, adaptive_block_size, adaptive_constant
        )
        adaptive_thresh_val = adaptive_constant
    
    # Apply morphological operations if requested
    if apply_morphology:
        if morph_kernel_size % 2 == 0:
            morph_kernel_size += 1  # Ensure odd size
            
        kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
        
        if morph_operation == "close":
            # Close operation fills small holes inside foreground objects
            shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, kernel, iterations=morph_iterations)
        elif morph_operation == "open":
            # Open operation removes small noise
            shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, kernel, iterations=morph_iterations)
        elif morph_operation == "dilate":
            # Dilate expands white regions
            shadow_mask = cv2.dilate(shadow_mask, kernel, iterations=morph_iterations)
        elif morph_operation == "erode":
            # Erode shrinks white regions
            shadow_mask = cv2.erode(shadow_mask, kernel, iterations=morph_iterations)
    
    # Convert original to RGB for display
    original_image_rgb = cv2.cvtColor(original_image_bgr, cv2.COLOR_BGR2RGB)
    
    return original_image_rgb, gray_original, shadow_mask, otsu_thresh_val, adaptive_thresh_val

def main():
    input_folder = "moon_images"
    # Changed output folder name to reflect new method
    output_folder = "moon_shadow_masks_adaptive" 

    # --- Parameters ---
    # Choose the detection method:
    # 'adaptive': Uses adaptive thresholding (best for uneven lighting)
    # 'otsu': Uses Otsu's method (simpler but less effective for shadows)
    # 'darkness': Uses percentage of darkest pixels
    # 'canny': Uses edge detection and contour filling
    DETECTION_METHOD = "canny"  
    
    # Enhance contrast before processing
    ENHANCE_CONTRAST = True
    
    # Parameters for adaptive thresholding
    ADAPTIVE_BLOCK_SIZE = 31  # Must be odd, larger values detect larger shadow regions
    ADAPTIVE_CONSTANT = 3     # Smaller values detect more shadows
    
    # Parameters for darkness-based method
    DARK_SHADOW_PERCENTAGE = 10  # Percentage of darkest pixels to consider as shadows
    
    # Apply morphological operations to clean up the mask
    APPLY_MORPHOLOGY = True
    MORPH_OPERATION = "dilate"   # 'dilate' expands shadow regions to connect nearby areas
    MORPH_KERNEL_SIZE = 5       # Size of the morphological kernel
    MORPH_ITERATIONS = 2        # Number of times to apply the operation
    # --- End Parameters ---

    if not os.path.isdir(input_folder):
        print(f"Error: Input folder '{input_folder}' not found.")
        return

    os.makedirs(output_folder, exist_ok=True)
    print(f"Output will be saved to '{output_folder}'.")
    print(f"Detection method: {DETECTION_METHOD}")
    if APPLY_MORPHOLOGY:
        print(f"Morphological {MORPH_OPERATION} will be applied with kernel={MORPH_KERNEL_SIZE}x{MORPH_KERNEL_SIZE}, iterations={MORPH_ITERATIONS}")
    else:
        print("Morphological post-processing is disabled.")

    processed_files_count = 0
    image_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]

    if not image_files:
        print(f"No files found in '{input_folder}'.")
        return

    print(f"Found {len(image_files)} files in '{input_folder}'. Attempting to process images...")

    for filename in image_files:
        image_path = os.path.join(input_folder, filename)

        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')):
            print(f"\nProcessing: {filename}...")

            original_image, gray_image, final_shadow_mask, otsu_t, adaptive_t = process_image_for_shadows(
                image_path,
                method=DETECTION_METHOD,
                enhance_contrast=ENHANCE_CONTRAST,
                dark_shadow_percentage=DARK_SHADOW_PERCENTAGE,
                adaptive_block_size=ADAPTIVE_BLOCK_SIZE,
                adaptive_constant=ADAPTIVE_CONSTANT,
                apply_morphology=APPLY_MORPHOLOGY,
                morph_operation=MORPH_OPERATION,
                morph_kernel_size=MORPH_KERNEL_SIZE,
                morph_iterations=MORPH_ITERATIONS
            )

            if final_shadow_mask is not None:
                base, ext = os.path.splitext(filename)
                method_tag = f"_{DETECTION_METHOD}"
                morph_tag = f"_{MORPH_OPERATION}" if APPLY_MORPHOLOGY else ""
                output_filename = f"{base}_shadow_mask{method_tag}{morph_tag}{ext}"
                output_path = os.path.join(output_folder, output_filename)

                try:
                    cv2.imwrite(output_path, final_shadow_mask)
                    print(f"Successfully saved shadow mask to: {output_path}")
                    processed_files_count += 1
                    
                    if processed_files_count == 2: # Display only the first one
                        plt.figure(figsize=(20, 5)) 
                        
                        plt.subplot(1, 4, 1)
                        plt.imshow(original_image)
                        plt.title('Original Image')
                        plt.axis('off')

                        plt.subplot(1, 4, 2)
                        plt.imshow(gray_image, cmap='gray')
                        plt.title('Grayscale Image')
                        plt.axis('off')
                        
                        # Show mask using Otsu's method for comparison
                        _, _, otsu_mask_for_display, _, _ = process_image_for_shadows(
                            image_path, method="otsu", enhance_contrast=False, 
                            apply_morphology=False
                        )
                        if otsu_mask_for_display is not None:
                            plt.subplot(1, 4, 3)
                            plt.imshow(otsu_mask_for_display, cmap='gray')
                            plt.title(f"Otsu Mask (Thresh: {otsu_t:.0f})")
                            plt.axis('off')
                        
                        plt.subplot(1, 4, 4) 
                        plt.imshow(final_shadow_mask, cmap='gray')
                        
                        if DETECTION_METHOD == "adaptive":
                            title = f"Adaptive Mask (B:{ADAPTIVE_BLOCK_SIZE}, C:{ADAPTIVE_CONSTANT})"
                        elif DETECTION_METHOD == "darkness":
                            title = f"Darkness Mask ({DARK_SHADOW_PERCENTAGE}% darkest)"
                        elif DETECTION_METHOD == "canny":
                            title = "Edge-based Mask"
                        else:
                            title = f"Final Mask (Method: {DETECTION_METHOD})"
                            
                        if APPLY_MORPHOLOGY: 
                            title += f" + {MORPH_OPERATION.capitalize()}"
                            
                        plt.title(title)
                        plt.axis('off')
                        
                        plt.suptitle(f'Example: {filename}', fontsize=16)
                        plt.tight_layout(rect=[0, 0, 1, 0.95])
                        plt.show()
                        
                except Exception as e:
                    print(f"Error saving shadow mask for {filename}: {e}")
            else:
                print(f"Skipping {filename} due to processing error.")
        else:
            print(f"Skipping non-image file: {filename}")

    print(f"\nFinished processing. {processed_files_count} shadow masks saved in '{output_folder}'.")

if __name__ == '__main__':
    main()
