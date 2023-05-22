# Resamples the M01 GeoTIFFs

# Platform: ntsgcompute22017.ntsg.umt.edu

# OUTPUT_DIR="/anx_lagr3/arthur.endsley/SMAP_L4C/ancillary_data/"
TEMP_DIR="/ntsg_home"

for i in 0 1 2; do
  echo "Resampling C${i}..."
  gdalwarp -overwrite -ts 3856 1624 -r average -srcnodata "-9999" -dstnodata "-9999"\
    $TEMP_DIR/tcf_OL7000_C${i}_M01_0002089.tiff $TEMP_DIR/tcf_OL7000_C${i}_M09_0002089.tiff
done

echo "Resampling annual NPP sum..."
gdalwarp -overwrite -ts 3856 1624 -r average -srcnodata "-9999" -dstnodata "-9999"\
  $TEMP_DIR/tcf_OL7000_npp_sum_M01.tiff $TEMP_DIR/tcf_OL7000_npp_sum_M09.tiff
