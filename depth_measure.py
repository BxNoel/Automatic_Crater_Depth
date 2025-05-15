import numpy as np
import math
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import get_body # Using get_body as it resolved a previous import issue for you
import astropy.units as u
import matplotlib.pyplot as plt
from astroquery.jplhorizons import Horizons # For JPL Horizons API query

# --- Function to get Sun's altitude from JPL Horizons API (Corrected) ---
import requests
from datetime import datetime, timedelta

def get_sun_altitude_from_lunar_site(lunar_lat_deg, lunar_lon_deg, observation_time_str):
    """
    Queries JPL Horizons API directly to get the Sun's altitude from a specific site on the Moon.
    
    Parameters:
    - lunar_lat_deg (float): Latitude on the Moon (degrees, North positive).
    - lunar_lon_deg (float): Longitude on the Moon (degrees, 0-360 East positive).
    - observation_time_str (str): Observation time in ISO format (e.g., '2025-03-08T23:39:36').
    
    Returns:
    - float: Sun's altitude in degrees, or None if an error occurs.
    """
    print(f"\nAttempting to query JPL Horizons for Sun altitude at:")
    print(f"  Lunar Lat: {lunar_lat_deg:.4f} deg, Lunar Lon: {lunar_lon_deg:.4f} deg (0-360 E)")
    print(f"  Observation Time: {observation_time_str}")
    
    try:
        # Format the time for the API
        dt = datetime.fromisoformat(observation_time_str.replace('Z', '+00:00'))
        start_time = dt.strftime("%Y-%b-%d %H:%M")  # Format: YYYY-MMM-DD HH:MM (e.g. 2025-Mar-08 23:39)
        
        # Create stop time 1 minute later
        stop_time = (dt + timedelta(minutes=1)).strftime("%Y-%b-%d %H:%M")
        
        # Build the API URL
        url = "https://ssd.jpl.nasa.gov/api/horizons.api"
        
        params = {
            'format': 'text',
            'COMMAND': "'10'",  # Sun
            'EPHEM_TYPE': 'OBSERVER',
            'CENTER': 'c@301',  # Coordinate center on Moon
            'COORD_TYPE': 'GEODETIC',
            'SITE_COORD': f"'{lunar_lon_deg},{lunar_lat_deg},0'",  # lon,lat,height
            'START_TIME': f"'{start_time}'",
            'STOP_TIME': f"'{stop_time}'",
            'STEP_SIZE': "'1m'",
            'QUANTITIES': "'4'",  # Elevation angle
        }
        
        # Make the API request
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            # Extract elevation from the response
            if "$$SOE" in response.text and "$$EOE" in response.text:
                data_section = response.text.split("$$SOE")[1].split("$$EOE")[0].strip()
                lines = data_section.strip().split('\n')
                if lines:
                    # Get the first line of data (corresponds to our requested time)
                    line = lines[0].strip()
                    # Split by whitespace and get the last value which is the elevation
                    parts = line.split()
                    if len(parts) >= 4:  # Ensure we have enough parts
                        try:
                            sun_altitude = float(parts[-1])  # Last element is the elevation
                            print(f"Successfully retrieved Sun altitude via API: {sun_altitude:.3f} degrees")
                            return sun_altitude
                        except ValueError:
                            print(f"ERROR: Could not convert elevation value to float: {parts[-1]}")
                    else:
                        print(f"ERROR: Data line doesn't have enough elements: {line}")
                else:
                    print("ERROR: No data lines found between $$SOE and $$EOE markers")
            
            # If we couldn't parse the elevation, print the response for debugging
            print("Could not extract elevation from API response. Full response:")
            print(response.text)
            return None
        else:
            print(f"ERROR: API request failed with status code {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"ERROR querying JPL Horizons API: {e}")
        return None


# --- Step 0: User Inputs & FITS File Information ---
# ==============================================================================
# TODO: USER - Update this with the path to YOUR FITS file
fits_file_path = '/Users/fabianzuluagazuluaga/Desktop/Columbia/Astronomy/Final Project/Automatic_Crater_Depth/observations/18_39_36/MoonPic1_00001.fits' # <<< REPLACE THIS (e.g., '/Users/fabian/images/MoonPic1_00001.fits')

# Physical pixel size on sensor (XPIXSZ from your FITS header in micrometers)
sensor_pixel_size_um = 3.79999995231628 # (This was from your FITS header: XPIXSZ)

# Telescope focal length in mm
telescope_focal_length_mm = 3910.0

# TODO: USER - Provide details for the crater you want to analyze:
# Using Copernicus as an example with the coordinates you provided.
# Change these for other craters you want to analyze.
chosen_crater_name = "Copernicus"
crater_latitude_deg = 9.6209      # Selenographic Latitude (North positive)
crater_longitude_deg = 339.9214   # Selenographic Longitude (0-360 East positive)
# ==============================================================================

# --- Load DATE-OBS from FITS Header ---
print("--- Initializing ---")
try:
    with fits.open(fits_file_path) as hdul_header_check:
        header = hdul_header_check[0].header
        date_obs_str = header.get('DATE-OBS')
        if not date_obs_str:
            print(f"ERROR: 'DATE-OBS' keyword not found in FITS header of {fits_file_path}")
            exit()
        print(f"Using DATE-OBS from FITS file: {date_obs_str}")
except FileNotFoundError:
    print(f"ERROR: FITS file not found at '{fits_file_path}'")
    exit()
except Exception as e:
    print(f"ERROR: Could not read FITS file or its header. {e}")
    exit()

# --- Get Sun's Altitude using the API ---
sun_altitude_at_crater_deg = get_sun_altitude_from_lunar_site(
    crater_latitude_deg,
    crater_longitude_deg,
    date_obs_str
)

if sun_altitude_at_crater_deg is None:
    print("Failed to get Sun altitude from JPL Horizons API. Cannot proceed.")
    exit()
print("-----------------------------------------")

# --- Step 1: Calculate Image Scale (km/pixel) ---
print("\n--- Step 1: Calculating Image Scale ---")
sensor_pixel_size_m = sensor_pixel_size_um * 1e-6
telescope_focal_length_m = telescope_focal_length_mm / 1000.0
angular_scale_rad_per_pixel = sensor_pixel_size_m / telescope_focal_length_m
arcseconds_per_radian = 206264.806247
angular_scale_arcsec_per_pixel = angular_scale_rad_per_pixel * arcseconds_per_radian
print(f"Calculated Angular Scale: {angular_scale_arcsec_per_pixel:.3f} arcseconds/pixel")

obs_time_obj = Time(date_obs_str, format='isot', scale='utc')
try:
    moon_ephem = get_body('moon', obs_time_obj)
    moon_distance_km = moon_ephem.distance.to_value(u.km)
    print(f"Distance to Moon on {obs_time_obj.iso}: {moon_distance_km:.2f} km")
except Exception as e:
    print(f"ERROR: Could not get Moon's distance using get_body. {e}")
    exit()
    
image_scale_km_per_pixel = angular_scale_rad_per_pixel * moon_distance_km
print(f"Calculated Image Scale on Moon: {image_scale_km_per_pixel:.4f} km/pixel")
print("-----------------------------------------")

# --- Step 2: Load Full Image Data ---
print(f"\n--- Step 2: Loading image data for {chosen_crater_name} ---")
try:
    with fits.open(fits_file_path) as hdul:
        image_data = hdul[0].data
    print(f"Successfully loaded FITS image data: {fits_file_path}")
except Exception as e:
    print(f"ERROR: Could not read FITS image data. {e}")
    exit()
print("---------------------------------------------------")

# --- Step 3: Manually Input Shadow Pixel Coordinates ---
print("\n--- Step 3: Manually Input Shadow Pixel Coordinates ---")
print("Use your FITS viewer (e.g., SAOImage DS9, ImageJ) to find the pixel coordinates.")
print("Ensure your viewer's coordinate system matches how the image is displayed (e.g., origin).")
print("For matplotlib with origin='lower', (0,0) is bottom-left.")

try:
    shadow_tip_x = float(input("Enter shadow TIP X pixel coordinate: "))
    shadow_tip_y = float(input("Enter shadow TIP Y pixel coordinate: "))
    shadow_base_x = float(input("Enter shadow BASE X pixel coordinate: "))
    shadow_base_y = float(input("Enter shadow BASE Y pixel coordinate: "))
except ValueError:
    print("Invalid input. Please enter numbers for coordinates. Exiting.")
    exit()

shadow_length_pixels = np.sqrt((shadow_tip_x - shadow_base_x)**2 + (shadow_tip_y - shadow_base_y)**2)
print(f"Calculated shadow length from input coordinates: {shadow_length_pixels:.2f} pixels")

# Optional: Display the image with the marked shadow for verification
print("\nDisplaying image with marked shadow based on your input coordinates.")
print("Close the plot window to continue with calculations.")
plt.figure(figsize=(10, 8))
low_percentile, high_percentile = np.percentile(image_data, [1, 99])
plt.imshow(image_data, cmap='gray', origin='lower', vmin=low_percentile, vmax=high_percentile)
plt.plot([shadow_tip_x, shadow_base_x], [shadow_tip_y, shadow_base_y], 'r-', lw=2, label='Measured Shadow')
plt.scatter([shadow_tip_x, shadow_base_x], [shadow_tip_y, shadow_base_y], c='red', s=50, zorder=5)
plt.text(shadow_tip_x + 5, shadow_tip_y + 5, " Tip", color="yellow", fontsize=9, bbox=dict(facecolor='black', alpha=0.5))
plt.text(shadow_base_x + 5, shadow_base_y + 5, " Base", color="yellow", fontsize=9, bbox=dict(facecolor='black', alpha=0.5))
plt.title(f"Shadow for {chosen_crater_name} (coordinates from your input)")
plt.xlabel("X pixel coordinate")
plt.ylabel("Y pixel coordinate")
plt.legend()
plt.colorbar(label="Pixel Value")
plt.show() # This will pause script execution until plot window is closed

print("-----------------------------------")

# --- Step 4: Convert Shadow Length to Kilometers ---
print("\n--- Step 4: Convert Shadow to Kilometers ---")
shadow_length_km = shadow_length_pixels * image_scale_km_per_pixel
print(f"Shadow length: {shadow_length_km:.3f} km")
print("-----------------------------------------")

# --- Step 5: Calculate Crater Depth ---
print("\n--- Step 5: Calculate Crater Depth ---")
sun_altitude_radians = math.radians(sun_altitude_at_crater_deg)

if sun_altitude_radians <= 1e-6:
    print(f"Sun angle ({sun_altitude_at_crater_deg:.3f} deg) is too low. Depth calculation unreliable.")
    crater_depth_km = float('nan')
elif abs(math.pi/2 - sun_altitude_radians) < 1e-6:
    print(f"Sun angle ({sun_altitude_at_crater_deg:.3f} deg) is at or near zenith. Shadow length likely near zero.")
    crater_depth_km = 0.0
else:
    crater_depth_km = shadow_length_km * math.tan(sun_altitude_radians)
    print(f"Estimated Depth for {chosen_crater_name}: {crater_depth_km:.3f} km")
print("-------------------------------------")

print("\nCalculation Complete.")
print("IMPORTANT NOTES:")
print("- This script requires an internet connection for the JPL Horizons API call.")
print("- Accuracy depends on precise inputs (crater lat/lon for API, shadow pixel coords), image scale.")
print("- Assumes the shadow falls on a relatively flat crater floor perpendicular to the wall.")