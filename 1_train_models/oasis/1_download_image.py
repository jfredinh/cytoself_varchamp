"""Download images and metadata for batch 7.

The process involves:
1. Querying the cpg index
2. Downloading files

"""  # noqa: INP001

# Imports
from pathlib import Path

import polars as pl
from cpgdata.utils import download_s3_files, parallel

DATA_PATH = "/home/jfredinh/projects/dl-space/cytoself_varchamp/data-oasis"
DATA_PATH_CPG = "/home/jfredinh/projects/dl-space/cytoself_varchamp/data-CPG-index"
DATA_PATH_IMAGE_CSV = DATA_PATH + "/image_csvs"
DATA_PATH_TIFF_IMAGES = DATA_PATH + "/tiff_images"

def main() -> None:
    """Download images and metadata.

    This function uses the cpg index to locate all DNA and GFP images and Image.csv files for batch 7.

    """

    index_dir = Path(DATA_PATH_CPG)
    index_files = list(index_dir.glob("*.parquet"))

    # Download images
    index_df = pl.scan_parquet(index_files)
    
    print(index_df)
    

    index_df = (
        index_df
        .filter(pl.col("dataset_id").eq("cpg0037-oasis"))
        #.filter(pl.col("batch_id") == "2024_01_23_Batch_7")
        .filter(pl.col("leaf_node").str.contains(".tiff"))
        .filter((pl.col("leaf_node").str.contains("-ch1")) | (pl.col("leaf_node").str.contains("-ch2")))
        .select("key")
        .collect()
    )

    download_keys = list(index_df.to_dict()["key"])
    # parallel(download_keys, download_s3_files, ["cellpainting-gallery",
    #                                             Path("DATA_PATH_TIFF_IMAGES")],
    #                                             jobs=20)


    # Download Image.csv
    index_df = pl.scan_parquet(index_files)

    index_df = (
        index_df
        .filter(pl.col("dataset_id").eq("cpg0037-oasis"))
        #.filter(pl.col("batch_id") == "2024_01_23_Batch_7")
        #.filter(pl.col("key").str.contains("assaydev"))
        #.filter(pl.col("leaf_node").str.contains("Image.csv"))
        #.select("key")
        .collect()
    )

    df_ = index_df.to_pandas()

    import pdb
    pdb.set_trace()


    download_keys = list(index_df.to_dict()["key"])
    parallel(download_keys, download_s3_files, ["cellpainting-gallery",
                                                Path("DATA_PATH_IMAGE_CSV")],
                                                jobs=20)

if __name__ == "__main__":
    main()
