
#ifndef SPARSE_LAND_DEF
#define SPARSE_LAND_DEF 1

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <time.h>
#include <limits.h>
#include <assert.h>

#include "uuta.h"

#define   MAIN_NAME    "MKGRID"
#define   VERS_TAG     "v0.0-2014-02-21T18:33:00-07:00, laj"

#define   LAND_R_FILE  "/anx_v2/laj/smap/code/landdomdef/output/MCD12Q1_M09land_row.uint16"
#define   LAND_C_FILE  "/anx_v2/laj/smap/code/landdomdef/output/MCD12Q1_M09land_col.uint16"

#define   LLAND9KM           1664040ul       /*Total grid cells in sparse 9km land domain*/    

#define   LDOM1KM          507233664ul       /*Total number of cells in EGv2 1km global domain*/
#define   NROW1KM                14616       /*Number of grid rows in EGv2 1km */
#define   NCOL1KM                34704       /*Number of grid cols in EGv2 1km */

#define   LDOM3KM           56359296ul       /*Total number of cells in EGv2 3km global domain*/
#define   NROW3KM                 4872       /*Number of grid rows in EGv2 3km */
#define   NCOL3KM                11568       /*Number of grid cols in EGv2 3km */

#define   LDOM9KM            6262144ul       /*Total number of cells in EGv2 9km global domain*/
#define   NROW9KM                 1624       /*Number of grid rows in EGv2 9km */
#define   NCOL9KM                 3856       /*Number of grid cols in EGv2 9km */

#define   NESTED_3KM_IN_9KM        3ul       /*number of 3km grid cells along one side of in nested 9km domain*/
#define   NESTED_3KM_IN_9KM_SQ     9ul       /*Total number of 3km grid cells in nested 9km domain*/

#define   NESTED_1KM_IN_9KM        9ul       /*number of 1km grid cells along one side of in nested 9km domain*/
#define   NESTED_1KM_IN_9KM_SQ    81ul       /*Total number of 1km grid cells in nested 9km domain*/

#define   MXSTRLEN                 500
#define   FILLVAL                -9999       /*Fill values*/        

// SERIALIZATION OFFSETS 
// 2D offsets: y is outermost, x is innermost (fastest varying) term
#define M_2D_B0(x,y,n_y) (((x)*n_y)+(y))

/*Serialization offset for nested grid within 9km 2D grid*/
/*indicies x,y correspond to 9km grid, while i,j is the nested cell reference*/
#define M_2D_N9_B0(x,y,i,j,nj, ny) (((x)*nj + (i))*ny + (y)*nj + (j))

typedef struct {

  unsigned short   *row;
  unsigned short   *col;

} spland_ref_struct;

int spland_load_9km_rc(spland_ref_struct *SPLAND);

void spland_deflate_9km(spland_ref_struct SPLAND, void *src_p, void *dest_p, const unsigned int dataType);
void spland_deflate_1km(spland_ref_struct SPLAND, void *src_p, void *dest_p, const unsigned int dataType);

void spland_inflate_9km(spland_ref_struct SPLAND, void *src_p, void *dest_p, const unsigned int dataType);
void spland_inflate_1km(spland_ref_struct SPLAND, void *src_p, void *dest_p, const unsigned int dataType);

#endif
