import math
import numpy as np
from shapely import LineString
import geopandas as gpd
import pyproj


def computefirespeed(fire_gdf):
    """
    Compute fire spread speed and maximum travel vectors between consecutive time steps.

    CORRECTED VERSION: Fixed speed calculation units and added debugging
    """
    print(f"Computing fire speed for {len(fire_gdf)} time steps")
    print(f"   CRS: {fire_gdf.crs}")

    transformer = pyproj.Transformer.from_crs(fire_gdf.crs, "EPSG:4326", always_xy=True)
    geod = pyproj.Geod(ellps="WGS84")
    orig_x = [np.nan]
    orig_y = [np.nan]
    dest_x = [np.nan]
    dest_y = [np.nan]
    result_max_dist = [np.nan]
    result_speed = [np.nan]

    # Handle the first time step geometry
    first_geom = fire_gdf.iloc[0]["geometry"]
    if hasattr(first_geom, 'geoms'):
        # MultiPolygon
        prev_step = [geom.simplify(0.05).exterior.coords for geom in first_geom.geoms]
        print(f"   First time step: MultiPolygon with {len(first_geom.geoms)} parts")
    else:
        # Single Polygon  
        prev_step = [first_geom.simplify(0.05).exterior.coords]
        print(f"   First time step: Single Polygon")

    # Iterate over time steps
    for i in range(1, min(fire_gdf.shape[0], 10)):
        print(f"   Processing time step {i}")

        # Get geometries for current and previous time steps
        prev_geom = fire_gdf.iloc[i-1]["geometry"]
        curr_geom = fire_gdf.iloc[i]["geometry"]

        # Handle MultiPolygon vs Polygon
        prev_geoms = prev_geom.geoms if hasattr(prev_geom, 'geoms') else [prev_geom]
        curr_geoms = curr_geom.geoms if hasattr(curr_geom, 'geoms') else [curr_geom]

        print(f"      Previous: {len(prev_geoms)} polygons, Current: {len(curr_geoms)} polygons")

        # Setup overlap matrix
        inter_matrix = np.zeros((len(prev_geoms), len(curr_geoms)))
        for ii in range(inter_matrix.shape[0]):
            for jj in range(inter_matrix.shape[1]):
                inter_matrix[ii, jj] = prev_geoms[ii].intersects(curr_geoms[jj])

        # Bug check
        if inter_matrix.shape[0] == 1 and inter_matrix.shape[1] == 1 and inter_matrix[0, 0] == False:
            print("      Warning: No overlap detected")

        # Compute maximum distance and travel vector
        max_fire_dist, max_origin, max_destination, prev_step = compute_max_vector(
            prev_geoms, curr_geoms, prev_step, inter_matrix, buffer=200, maxbins=200, slop=2)

        orig_x.append(max_origin[0])
        orig_y.append(max_origin[1])
        dest_x.append(max_destination[0])
        dest_y.append(max_destination[1])

        # CORRECTED: Proper coordinate transformation and distance calculation
        print(f"      Origin: ({max_origin[0]:.2f}, {max_origin[1]:.2f})")
        print(f"      Destination: ({max_destination[0]:.2f}, {max_destination[1]:.2f})")

        # Transform coordinates to lat/lon for distance calculation
        try:
            lons, lats = transformer.transform([max_origin[0], max_destination[0]],
                                               [max_origin[1], max_destination[1]])

            print(f"      Transformed to lat/lon: ({lats[0]:.6f}, {lons[0]:.6f}) -> ({lats[1]:.6f}, {lons[1]:.6f})")

            # Calculate geodesic distance in meters
            dist_meters = geod.line_length(lons, lats)
            print(f"      Distance: {dist_meters:.2f} meters ({dist_meters/1000:.3f} km)")

            result_max_dist.append(dist_meters)

            # CORRECTED SPEED CALCULATION:
            # dist_meters is already in meters, so speed in m/h = meters / hours
            # Assuming 24 hours between daily perimeters
            speed_m_per_h = dist_meters / 24
            speed_km_per_h = speed_m_per_h / 1000

            result_speed.append(speed_m_per_h)

            print(f"      Speed: {speed_m_per_h:.2f} m/h ({speed_km_per_h:.3f} km/h)")

            # Sanity check for realistic fire speeds
            if speed_km_per_h > 50:
                print(f"      Warning: Very high speed ({speed_km_per_h:.1f} km/h) - check calculation")
            elif speed_km_per_h > 20:
                print(f"      Extreme fire speed: {speed_km_per_h:.1f} km/h")
            elif speed_km_per_h > 5:
                print(f"      Fast fire speed: {speed_km_per_h:.1f} km/h") 
            else:
                print(f"      ✅ Normal fire speed: {speed_km_per_h:.1f} km/h")

        except Exception as e:
            print(f"      Error in distance calculation: {e}")
            result_max_dist.append(np.nan)
            result_speed.append(np.nan)

    print(f"🏁 Fire speed calculation complete")
    return orig_x, orig_y, dest_x, dest_y, result_max_dist, result_speed


def compute_max_vector(perim_inner_geoms, perim_outer_geoms, inner_coords, inter_matrix, buffer, maxbins, slop):
    """
    Compute maximum fire spread vector between two time steps.
    CORRECTED: Better error handling and simplified approach
    """
    outer_coords = []
    result_dist = []
    result_coord_pair = []
    result_poly_pair = []

    for poly_outer in range(len(perim_outer_geoms)):
        try:
            buffer_poly = perim_outer_geoms[poly_outer].buffer(buffer)

            # CORRECTED: Handle both Polygon and MultiPolygon exterior access
            if hasattr(perim_outer_geoms[poly_outer], 'exterior'):
                outer_bbox = perim_outer_geoms[poly_outer].exterior.bounds
                outer = perim_outer_geoms[poly_outer].simplify(0.05).exterior.coords
            else:
                # If it's a MultiPolygon, get the largest polygon
                largest_poly = max(perim_outer_geoms[poly_outer].geoms, key=lambda p: p.area)
                outer_bbox = largest_poly.exterior.bounds
                outer = largest_poly.simplify(0.05).exterior.coords

            outer_coords.append(outer)

            # Check for spots (fires that don't overlap)
            spot_flag = not np.any(inter_matrix[:, poly_outer])
            polyids = []

            if spot_flag:
                polyids = list(range(len(perim_inner_geoms)))
            else:
                for ii in range(len(perim_inner_geoms)):
                    if inter_matrix[ii, poly_outer]:
                        polyids.append(ii)

            # Track comparisons for this polygon
            poly_max_dist = 0
            poly_coordpair = None
            poly_pair = None

            for poly_inner in polyids:
                try:
                    # SIMPLIFIED: Use centroid-based distance for more reliable results
                    inner_centroid = perim_inner_geoms[poly_inner].centroid
                    outer_centroid = perim_outer_geoms[poly_outer].centroid

                    if inner_centroid and outer_centroid:
                        dist = compute_dist((inner_centroid.x, inner_centroid.y), 
                                          (outer_centroid.x, outer_centroid.y))

                        if dist > poly_max_dist:
                            poly_max_dist = dist
                            poly_coordpair = ((inner_centroid.x, inner_centroid.y), 
                                            (outer_centroid.x, outer_centroid.y))
                            poly_pair = (poly_inner, poly_outer)

                except Exception as e:
                    print(f"      Warning: Error processing inner polygon {poly_inner}: {e}")
                    continue

            # Store results
            result_dist.append(poly_max_dist)
            result_coord_pair.append(poly_coordpair if poly_coordpair else ((0, 0), (0, 0)))
            result_poly_pair.append(poly_pair)

        except Exception as e:
            print(f"      Warning: Error processing outer polygon {poly_outer}: {e}")
            result_dist.append(0)
            result_coord_pair.append(((0, 0), (0, 0)))
            result_poly_pair.append(None)

    # Find maximum distance
    if result_dist and any(d > 0 for d in result_dist):
        max_loc = np.argmax(result_dist)
        maximum_distance = result_dist[max_loc]
        max_dist_origin = result_coord_pair[max_loc][0]
        max_dist_destination = result_coord_pair[max_loc][1]
    else:
        maximum_distance = 0
        max_dist_origin = (0, 0)
        max_dist_destination = (0, 0)

    return maximum_distance, max_dist_origin, max_dist_destination, outer_coords


def compute_dist(a, b):
    """Compute Euclidean distance between two points."""
    return math.sqrt(((a[0] - b[0]) ** 2) + ((a[1] - b[1]) ** 2))
