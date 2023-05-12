# Inflates and the initial M01 SOC state

# Platform: ntsgcompute22017.ntsg.umt.edu

MKGRID="/anx_lagr4/SMAP/L4C_code/utils/mkgrid/mkgrid"
LOGFILE="/anx_lagr4/SMAP/L4C_code/tcf/log/mkgrid"
# OUTPUT_DIR="/anx_lagr3/arthur.endsley/SMAP_L4C/ancillary_data/"
TEMP_DIR="/home/arthur.endsley"

for i in 0 1 2; do
  echo "Inflating C${i}..."
  $MKGRID inflate $LOGFILE $TEMP_DIR /anx_lagr4/SMAP/L4C_code/tcf/output/OL7000/tcf_OL7000_C${i}_M01land_0002089.flt32
done
