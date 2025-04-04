"""Filter cells and crop image.

For each allele:
1. Filter cells for QA/QC
2. Extract 100x100 pixel crop for AGP and dapi
3. Save as stacked numpy array

"""  # noqa: INP001

import os

import numpy as np
import polars as pl
import zarr
from tqdm import tqdm
import pandas as pd


def crop_allele(cmp: str, profile_df: pl.DataFrame, img_dir: str, out_dir: str) -> None:
    """Crop images and save metadata as numpy arrays for one allele.

    Parameters
    ----------
    allele : String
        Name of allele to process
    profile_df : String
        Dataframe with pathname and cell coordinates
    img_dir : String
        Directory where all images are stored
    out_dir : String
        Directory where numpy arrays should be saved

    """
    # allele_df = profile_df.filter(pl.col("Compound_label") == allele)
    # sites = allele_df.select("Metadata_SiteID").to_series().unique().to_list()

    compound_df = profile_df[profile_df["Compound_label"] == cmp]
    sites = list(compound_df["Metadata_SiteID"].unique())

    meta = []
    AGP = []
    dna = []

    for site in sites:
        # site_df = allele_df.filter(pl.col("Metadata_SiteID") == site)
        site_df = compound_df[compound_df.Metadata_SiteID == site]

        # meta.append(site_df.select([
        #     "Compound_label",
        #     "Metadata_CellID",
        # ]))

        meta.append(site_df[[
            "Compound_label",
            "Metadata_CellID",
        ]])


        dna_zarr = site_df["DNA_zarrpath"].unique()[0]
        AGP_zarr = site_df["AGP_zarrpath"].unique()[0]
        AGP_path = f"{AGP_zarr}"
        dna_path = f"{dna_zarr}"

        AGP_img = zarr.open_group(AGP_path, mode='r')
        dna_img = zarr.open_group(dna_path, mode='r')
        #for row in site_df.iter_rows(named=True):

        import pdb
        pdb.set_trace()
        for i, row in site_df.iterrows():
            x1, x2 = row["x_low"], row["x_high"]
            y1, y2 = row["y_low"], row["y_high"]

            AGP.append(AGP_img[y1:y2, x1:x2])
            dna.append(dna_img[y1:y2, x1:x2])

    # Stack and save arrays
    AGP_array = np.stack(AGP)
    dna_array = np.stack(dna)
    meta_array = pl.concat(meta).to_numpy()

    np.save(f"{out_dir}/{cmp}_label.npy", meta_array)
    np.save(f"{out_dir}/{cmp}_pro.npy", AGP_array)
    np.save(f"{out_dir}/{cmp}_nuc.npy", dna_array)


def main() -> None:
    """Filter cells and crop data.

    Filter all cells according to many QA/QC criteria and then crop cells.

    """
    # Paths
    imagecsv_dir = f"/home/jfredinh/projects/dl-space/cytoself_varchamp/1_train_models/oasis/DATA_PATH_IMAGE_CSV/prod_25"
    prof_path = f"/home/jfredinh/projects/dl-space/cytoself_varchamp/data-oasis/profiles/"
    meta_path = f"/home/jfredinh/projects/dl-space/cytoself_varchamp/data-oasis/metadata/"
    img_dir = f"/home/jfredinh/projects/dl-space/cytoself_varchamp/data-oasis/zarr_images"
    out_dir = "/home/jfredinh/projects/dl-space/cytoself_varchamp/data-oasis/single_cell"

    prof_dir = f"/home/jfredinh/projects/dl-space/cytoself_varchamp/data-oasis/nucleus_data.csv"

    # Filter thresholds
    min_area_ratio = 0.15
    max_area_ratio = 0.3
    min_center = 50
    max_center = 1030
    num_mad = 5
    min_cells = 250

    suffixes = ["/plate_41002688/plate_41002688.csv",
                "/plate_41002689/plate_41002689.csv"]
    
    suffixes_meta = ["/plate_41002688/metadata.parquet",
                "/plate_41002689/metadata.parquet"]
    
    suffixes_bioc = ["/plate_41002688/biochem.parquet",
                "/plate_41002689/biochem.parquet"]
    
    full_columns_list = ["Metadata_well_position", "Metadata_ImageNumber", 
                         "Metadata_ObjectNumber", "Metadata_symbol", "compound_name", 
                         "Metadata_control_type", "Metadata_Plate",
                        "Nuclei_AreaShape_Area", "Cells_AreaShape_Area", 
                        "Nuclei_AreaShape_Center_X", "Nuclei_AreaShape_Center_Y",
                        "Cells_Intensity_MedianIntensity_AGP", "Cells_Intensity_IntegratedIntensity_AGP"]
    
    full_columns_list = ["Metadata_well_position",  "Metadata_ImageNumber", 
                         "Metadata_ObjectNumber", "Metadata_symbol", "compound_name", 
                         "Metadata_control_type", "Metadata_Plate",
                        "Nuclei_AreaShape_Area", "Cells_AreaShape_Area", 
                        "Nuclei_AreaShape_Center_X", "Nuclei_AreaShape_Center_Y",
                        "Cells_Intensity_MedianIntensity_AGP", "Cells_Intensity_IntegratedIntensity_AGP"]

    full_columns_list = ["Metadata_Well",  "compound_name", 
                        "Metadata_Plate",
                        "Nuclei_AreaShape_Area", "Cells_AreaShape_Area", 
                        "Nuclei_AreaShape_Center_X", "Nuclei_AreaShape_Center_Y",
                        "Cells_Intensity_MedianIntensity_AGP", "Cells_Intensity_IntegratedIntensity_AGP"]

    combined      = pd.concat([pd.read_csv(prof_path+f) for f in suffixes])
    #combined_meta = pd.concat([pd.read_parquet(meta_path+f) for f in suffixes_meta])
    combined_bioc = pd.concat([pd.read_parquet(meta_path+f) for f in suffixes_bioc])

    profiles = combined.merge(combined_bioc, 
                                   how="left", 
                                   left_on=["Metadata_Plate", "Metadata_Well"], 
                                   right_on=["plate", "well"])

    profiles = pd.read_csv(prof_dir, index_col=0)
    # # Get metadata
    # profiles = [pl.scan_csv(prof_path+f).select(
    #     ["Metadata_well_position", "Metadata_ImageNumber", "Metadata_ObjectNumber",
    #      "Metadata_symbol", "compound_name", "Metadata_control_type", "Metadata_Plate",
    #      "Nuclei_AreaShape_Area", "Cells_AreaShape_Area", "Nuclei_AreaShape_Center_X", "Nuclei_AreaShape_Center_Y",
    #      "Cells_Intensity_MedianIntensity_AGP", "Cells_Intensity_IntegratedIntensity_AGP"],
    # ).collect() for f in suffixes ]

    # # Filter based on cell to nucleus area
    # profiles = profiles.with_columns(
    #                 (pl.col("Nuclei_AreaShape_Area")/pl.col("Cells_AreaShape_Area")).alias("Nucleus_Cell_Area"),
    #                 pl.concat_str([
    #                     "Metadata_Plate", "Metadata_well_position", "Metadata_ImageNumber", "Metadata_ObjectNumber",
    #                     ], separator="_").alias("Metadata_CellID"),
    #         ).filter((pl.col("Nucleus_Cell_Area") > min_area_ratio) & (pl.col("Nucleus_Cell_Area") < max_area_ratio))

    # # Filter cells too close to image edge
    # profiles = profiles.filter(
    #     ((pl.col("Nuclei_AreaShape_Center_X") > min_center) & (pl.col("Nuclei_AreaShape_Center_X") < max_center) &
    #     (pl.col("Nuclei_AreaShape_Center_Y") > min_center) & (pl.col("Nuclei_AreaShape_Center_Y") < max_center)),
    # )



    #profiles = profiles[((profiles["Nuclei_AreaShape_Area"]/profiles["Cells_AreaShape_Area"]) > min_area_ratio) &
    #                    ((profiles["Nuclei_AreaShape_Area"]/profiles["Cells_AreaShape_Area"]) < max_area_ratio) ]

    # profiles = profiles[((profiles["Nuclei_AreaShape_Center_X"] > min_center) & (profiles["Nuclei_AreaShape_Center_X"]) < max_center) &
    #                     ((profiles["Nuclei_AreaShape_Center_Y"] > min_center) & (profiles["Nuclei_AreaShape_Center_Y"]) < max_center)]


    profiles = profiles[((profiles["AreaShape_Center_X"] > min_center) & (profiles["AreaShape_Center_X"]) < max_center) &
                        ((profiles["AreaShape_Center_Y"] > min_center) & (profiles["AreaShape_Center_Y"]) < max_center)]

    # Calculate median and mad of AGP intensity for each allele
    #medians = profiles.group_by(["Metadata_Plate", "Metadata_well_position"]).agg(
    #    pl.col("Cells_Intensity_MedianIntensity_AGP").median().alias("WellIntensityMedian"),
    #)

#     medians = profiles.groupby(["Metadata_Plate", "Metadata_well_position"]).Cells_Intensity_MedianIntensity_AGP.median()
# )

#     profiles = profiles.join(medians, on=["Metadata_Plate", "Metadata_well_position"])

#     profiles = profiles.with_columns(
#         (pl.col("Cells_Intensity_MedianIntensity_AGP") - pl.col("WellIntensityMedian")).abs().alias("Abs_dev"),
#     )
#     mad = profiles.group_by(["Metadata_Plate", "Metadata_well_position"]).agg(
#         pl.col("Abs_dev").median().alias("Intensity_MAD"),
#     )
#     profiles = profiles.join(mad, on=["Metadata_Plate", "Metadata_well_position"])

#     # Threshold is 5X
#     profiles = profiles.with_columns(
#         (pl.col("WellIntensityMedian") + num_mad*pl.col("Intensity_MAD")).alias("Intensity_upper_threshold"),
#         (pl.col("WellIntensityMedian") - num_mad*pl.col("Intensity_MAD")).alias("Intensity_lower_threshold"),
#     )

#     # Filter by intensity MAD
#     profiles = profiles.filter(
#         pl.col("Cells_Intensity_MedianIntensity_AGP") <= pl.col("Intensity_upper_threshold"),
#     ).filter(
#         pl.col("Cells_Intensity_MedianIntensity_AGP") >= pl.col("Intensity_lower_threshold"),
#     )

    # Filter out allele set 5 (mismatched metadata)
    #profiles = profiles.filter(pl.col("Metadata_plate_map_name") != "B7A2R1_P1")

    # Filter out alleles with fewer than 250 cells
    # keep_alleles = profiles.group_by("compound_name").count().filter(
    #     pl.col("count") >= min_cells,
    #     ).select("compound_name").to_series().to_list()
    # profiles = profiles.filter(pl.col("compound_name").is_in(keep_alleles))

    # keep_alleles = profiles.groupby("compound_name").count().filter(
    #     pl.col("count") >= min_cells,
    #     ).select("compound_name").to_series().to_list()
    # profiles = profiles.filter(pl.col("compound_name").is_in(keep_alleles))


    # add full crop coordinates
    # profiles = profiles.with_columns(
    #     (pl.col("Nuclei_AreaShape_Center_X") - 50).alias("x_low").round().cast(pl.Int16),
    #     (pl.col("Nuclei_AreaShape_Center_X") + 50).alias("x_high").round().cast(pl.Int16),
    #     (pl.col("Nuclei_AreaShape_Center_Y") - 50).alias("y_low").round().cast(pl.Int16),
    #     (pl.col("Nuclei_AreaShape_Center_Y") + 50).alias("y_high").round().cast(pl.Int16),
    # )


    profiles["x_low"]  = (profiles.AreaShape_Center_X - 50).astype(np.int16)
    profiles["x_high"] = (profiles.AreaShape_Center_X + 50).astype(np.int16)
    profiles["y_low"]  = (profiles.AreaShape_Center_Y - 50).astype(np.int16)
    profiles["y_high"] = (profiles.AreaShape_Center_Y + 50).astype(np.int16)



    # Read in all Image.csv to get ImageNumber:SiteNumber mapping and paths
    # image_dat = []
    # icf = os.listdir(imagecsv_dir)
    # for fp in tqdm(icf):
    #     plate, well = fp.split("-")

    #     image_dat.append(pl.read_csv(f"{imagecsv_dir}/{fp}/Image.csv").select(
    #         [
    #             "ImageNumber",
    #             "Metadata_Site",
    #             "PathName_OrigDNA",
    #             "FileName_OrigDNA",
    #             "FileName_OrigAGP",
    #             ],
    #         ).with_columns(
    #         pl.lit(plate).alias("Metadata_Plate"),
    #         pl.lit(well).alias("Metadata_well_position"),
    #         ))
    # image_dat = pl.concat(image_dat).rename({"ImageNumber": "Metadata_ImageNumber"})

    profiles = profiles.merge(combined_bioc, 
                                   how="left", 
                                   left_on=["Metadata_Plate", "Metadata_well_position"], 
                                   right_on=["plate", "well"])

    profiles["DNA_zarrpath"] = img_dir + "/" + profiles.Metadata_Plate + "/" + profiles.FileName_OrigDNA
    profiles["AGP_zarrpath"] = img_dir + "/" + profiles.Metadata_Plate + "/" + profiles.FileName_OrigAGP


    # # Create useful filepaths
    # image_dat = image_dat.with_columns(
    #     pl.col("PathName_OrigDNA").str.replace(".*cpg0020-varchamp/", "").alias("Path_root"),
    # )
    # image_dat = image_dat.with_columns(
    #     pl.concat_str(["Path_root", "FileName_OrigDNA"], separator="/").str.replace(
    #         "tiff", "zarr").alias("DNA_zarrpath"),
    #     pl.concat_str(["Path_root", "FileName_OrigAGP"], separator="/").str.replace(
    #         "tiff", "zarr").alias("AGP_zarrpath"),
    # )
    
    

    # image_dat = image_dat.drop([
    #     "PathName_OrigDNA",
    #     "FileName_OrigDNA",
    #     "FileName_OrigAGP",
    #     "Path_root",
    # ])

    # Append to profiles
    # profiles = profiles.join(image_dat, on = ["Metadata_Plate", "Metadata_well_position", "Metadata_ImageNumber"])

    # Sort by compound_name, then image number
    # profiles = profiles.with_columns(
    #     pl.concat_str(["Metadata_Plate", "Metadata_well_position", "Metadata_Site"], separator="_").alias(
    #         "Metadata_SiteID"),
    #     pl.col("compound_name").str.replace("_", "-").alias("Protein_label"),
    # )

    profiles["Compound_label"] = profiles.compound_name.replace("_", "-")
    profiles["Metadata_SiteID"] = profiles.Metadata_Plate.astype(str) + "_" + profiles.Metadata_well_position.astype(str) + "_" + profiles.Metadata_Site.astype(str)
    profiles["Metadata_CellID"] = profiles.Metadata_Plate.astype(str) + "_" + profiles.Metadata_well_position.astype(str) + "_" + profiles.Metadata_Site.astype(str) + "_" + profiles.ObjectNumber.astype(str)

    # profiles = profiles.with_columns(
    #     pl.concat_str(["Metadata_Plate", "Metadata_well_position", "Metadata_Site"], separator="_").alias(
    #         "Metadata_SiteID"),
    #     pl.col("compound_name").str.replace("_", "-").alias("Protein_label"),
    # )


    # profiles = profiles.sort(["Protein_label", "Metadata_SiteID"])
    # alleles = profiles.select("Protein_label").to_series().unique().to_list()


    profiles  = profiles.sort_values(["Compound_label", "Metadata_SiteID"])
    compounds = list(profiles["Compound_label"].unique())

    for cmp in tqdm(compounds):
        crop_allele(cmp, profiles, img_dir, out_dir)


if __name__ == "__main__":
    main()
