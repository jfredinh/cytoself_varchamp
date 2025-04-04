"""Convert tiff to zarr.

For each image:
1. Rescale intensity value
2. Save as .zarr

"""  # noqa: INP001

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

import numpy as np
import polars as pl
import tifffile as tiff
import zarr
from skimage import exposure
from tqdm import tqdm

import pandas as pd

def get_img_paths(imagecsv_dir: str) -> pl.DataFrame:
    """Process individual Image.csv into dataframe.

    Parameters
    ----------
    imagecsv_dir : String
        Directory where all Image.csv are stored

    Returns
    -------
    pl.DataFrame
        Dataframe containing all concatenated Image.csv files.

    """
    image_dat = []
    icf = os.listdir(imagecsv_dir)
    for fp in tqdm(icf):
        plate, well = fp.split("-")
        if plate not in ["plate_41002688", "plate_41002689"]:
            continue
        if os.path.exists(f"{imagecsv_dir}/{fp}/Image.csv"):
            image_dat.append(pl.read_csv(f"{imagecsv_dir}/{fp}/Image.csv").select(
                [
                    "ImageNumber",
                    "Metadata_Site",
                    "PathName_OrigDNA",
                    "FileName_OrigDNA",
                    "FileName_OrigAGP"
                    #"FileName_OrigGFP",
                    ],
                ).with_columns(
                pl.lit(plate).alias("Metadata_Plate"),
                pl.lit(well).alias("Metadata_well_position"),
                ))
    return pl.concat(image_dat).rename({"ImageNumber": "Metadata_ImageNumber"})



def get_nuclei_centers(imagecsv_dir: str) -> pl.DataFrame:
    """Process individual Image.csv into dataframe.

    Parameters
    ----------
    imagecsv_dir : String
        Directory where all Image.csv are stored

    Returns
    -------
    pl.DataFrame
        Dataframe containing all concatenated Image.csv files.

    """
    image_dat = []

    icf = os.listdir(imagecsv_dir)
    for fp in tqdm(icf):
        
        plate, well, site = fp.split("-")
        
        if plate not in ["plate_41002688", "plate_41002689"]:
            continue
        if os.path.exists(f"{imagecsv_dir}/{fp}/Nuclei.csv") & os.path.exists(f"{imagecsv_dir}/{fp}/Image.csv"):
            

            #df_ = pd.read_csv(f"{imagecsv_dir}/{fp}/Image.csv")
            #site = df_[["Metadata_Site", "ImageNumber"]]#.unique()[0]

            well_level_img = pl.read_csv(f"{imagecsv_dir}/{fp}/Image.csv").select(
                [
                        "ImageNumber", 
                        "Metadata_Site",
                        "PathName_OrigDNA",
                        "FileName_OrigDNA",
                        "FileName_OrigAGP",
                        
                    ],
                )
            
            well_level_img = well_level_img.with_columns([
                    pl.col("ImageNumber").cast(pl.Int32),
                    pl.col("Metadata_Site").cast(pl.Int32)
                ])
                
            
            well_level = pl.read_csv(f"{imagecsv_dir}/{fp}/Nuclei.csv").select(
                [
                    "ImageNumber",
                    "Location_Center_X", 
                    "Location_Center_Y",
                    "AreaShape_Center_X",
                    "AreaShape_Center_Y",
                    "AreaShape_Area",
                    "ObjectNumber"
                    ],
                ).with_columns(
                pl.lit(site).alias("Metadata_Site"),
                pl.lit(plate).alias("Metadata_Plate"),
                pl.lit(well).alias("Metadata_well_position"),
                )
            
            well_level = well_level.with_columns([
                    pl.col("ImageNumber").cast(pl.Int32),
                    pl.col("ObjectNumber").cast(pl.Int32),
                    pl.col("Location_Center_X").cast(pl.Float64),
                    pl.col("Location_Center_Y").cast(pl.Float64),
                    pl.col("AreaShape_Center_X").cast(pl.Float64),
                    pl.col("AreaShape_Center_Y").cast(pl.Float64),
                    pl.col("AreaShape_Area").cast(pl.Float64),
                    pl.col("Metadata_Site").cast(pl.Utf8),
                    pl.col("Metadata_Plate").cast(pl.Utf8),
                    pl.col("Metadata_well_position").cast(pl.Utf8),
                ])

            # import pdb
            # pdb.set_trace()

            #combined = well_level.merge(well_level_img, on="ImageNumber", how="left")
            combined = well_level.join(well_level_img, on="ImageNumber")

            image_dat.append(combined)
            
    return pl.concat(image_dat).rename({"ImageNumber": "Metadata_ImageNumber"})

def tiff2zarr(tiffpath: str) -> None:
    """Rescales and converts tiff.

    Parameters
    ----------
    tiffpath : String
        Path to the tiff file to rescale and convert.

    """
    
    tiffpath = tiffpath.replace("axiom/images/prod_25/images/", "")
    tiffpath = f"/home/jfredinh/projects/dl-space/cytoself_varchamp/data-oasis/tiff_images/{tiffpath}"
    # Read in image
    
    img = tiff.imread(tiffpath)

    # Rescale from 0 to 1, at 99th percentile
    vmin = np.percentile(img, 00.5)
    vmax = np.percentile(img, 99.5)
    
    img = exposure.rescale_intensity(img, in_range=(vmin, vmax), out_range=(0, 1))

    # Save as zarr
    zarrpath = tiffpath.replace(".tiff", ".zarr")
    zarrpath = zarrpath.replace("tiff_images", "zarr_images")

    
    #zarr_array = zarr.array(img)

    #print(zarr_array)
    #print(type(zarr_array))

    zarr.save(zarrpath, img)

def run_in_parallel(function: Callable[[Any], None], args_list: list, max_workers: int) -> None:
    """Run function in parallel.

    Parameters
    ----------
    function : Function
        Function to execute in parallel.
    args_list : List
        List of arguments for the function
    max_workers : int
        Number of processes to launch

    """
    print("STARTING THREADPOOL")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks to the executor
        futures = {executor.submit(function, arg): arg for arg in args_list}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            future.result()


def main() -> None:
    """Rescale and save as zarr.

    This function reads in all tiff files, rescales from 0 to 99th percentils, and saves as zarr.

    """
    varchamp_dir = "/dgx1nas1/storage/data/jess/cytoself/varchamp_data"
    imagecsv_dir = f"/home/jfredinh/projects/dl-space/cytoself_varchamp/1_train_models/oasis/DATA_PATH_IMAGE_CSV/prod_25"
    imagecsv_dir_site = f"/home/jfredinh/projects/dl-space/cytoself_varchamp/data-oasis/csv-data/prod_25"

    n_thread = 1

    image_dat = get_img_paths(imagecsv_dir=imagecsv_dir)
    #nuclei_data = get_nuclei_centers(imagecsv_dir=imagecsv_dir_site)
    
    #nuclei_data = nuclei_data.to_pandas()
    #nuclei_data.to_csv("/home/jfredinh/projects/dl-space/cytoself_varchamp/data-oasis/nucleus_data.csv")

    # Create useful filepaths
    image_dat = image_dat.with_columns(
        pl.col("PathName_OrigDNA").str.replace(".*cpg0037-oasis/", "").alias("Path_root"),
    )
    image_dat = image_dat.with_columns(
        pl.concat_str(["Path_root", "FileName_OrigDNA"], separator="/").alias("DNA_imgpath"),
        pl.concat_str(["Path_root", "FileName_OrigAGP"], separator="/").alias("GFP_imgpath"),
    )

    image_dat = image_dat.drop([
        "PathName_OrigDNA",
        "FileName_OrigDNA",
        "FileName_OrigAGP",
        "Path_root",
    ])

    dna_path = image_dat.select("DNA_imgpath").to_series().unique().to_list()
    gfp_path = image_dat.select("GFP_imgpath").to_series().unique().to_list()
    img_paths = dna_path + gfp_path

    #img_paths = img_paths[:500]
    
    run_in_parallel(tiff2zarr, img_paths, max_workers=n_thread)


if __name__ == "__main__":
    main()
