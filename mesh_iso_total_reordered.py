import pygmsh
from pygmsh.geo import Geometry
import numpy as np
import matplotlib.pyplot as plt

def generate_wall_with_hole(L=6.0, H=4.0, R=1.0,
                            lc_outer=1, lc_inner=0.8):
    """
    Rectangle [0,L]×[0,H] minus a circle of radius R at its center,
    meshed with 6-node (order=2) triangles.
    """
    with pygmsh.geo.Geometry() as geom:
        # 1) Outer rectangle
        p1 = geom.add_point([0.0, 0.0, 0.0], mesh_size=lc_outer)
        p2 = geom.add_point([  L, 0.0, 0.0], mesh_size=lc_outer)
        p3 = geom.add_point([  L,   H, 0.0], mesh_size=lc_outer)
        p4 = geom.add_point([0.0,   H, 0.0], mesh_size=lc_outer)
        l1 = geom.add_line(p1, p2)
        l2 = geom.add_line(p2, p3)
        l3 = geom.add_line(p3, p4)
        l4 = geom.add_line(p4, p1)
        outer_loop = geom.add_curve_loop([l1, l2, l3, l4])

        # 2) Four points on the circle
        center = [L/2, H/2, 0.0]
        c0   = geom.add_point([center[0] + R, center[1]    , 0.0], mesh_size=lc_inner)
        c90  = geom.add_point([center[0]    , center[1] + R, 0.0], mesh_size=lc_inner)
        c180 = geom.add_point([center[0] - R, center[1]    , 0.0], mesh_size=lc_inner)
        c270 = geom.add_point([center[0]    , center[1] - R, 0.0], mesh_size=lc_inner)
        cen  = geom.add_point(center, mesh_size=lc_inner)

        # 3) Four circle-arcs
        a1 = geom.add_circle_arc(c0,   cen, c90)
        a2 = geom.add_circle_arc(c90,  cen, c180)
        a3 = geom.add_circle_arc(c180, cen, c270)
        a4 = geom.add_circle_arc(c270, cen, c0)
        hole_loop = geom.add_curve_loop([a1, a2, a3, a4])

        # 4) Define the domain as rectangle minus hole and mesh it
        geom.add_plane_surface(outer_loop, holes=[hole_loop])
        mesh = geom.generate_mesh(order=2)
        
        points = mesh.points[:, :2]
        tris6  = mesh.get_cells_type("triangle6")
        tris6_reordered = tris6[:,[0,1,2,4,5,3]]

    return points, tris6_reordered


def generate_wall_with_hole_fixed_elements(L=6.0, H=4.0, R=1.0, target_elements=500):
    """
    Generate a wall with a circular hole aiming for approximately target_elements.
    """
    
    def mesh_with_size(lc_outer, lc_inner):
        with pygmsh.geo.Geometry() as geom:
            p1 = geom.add_point([0.0, 0.0, 0.0], mesh_size=lc_outer)
            p2 = geom.add_point([L, 0.0, 0.0], mesh_size=lc_outer)
            p3 = geom.add_point([L, H, 0.0], mesh_size=lc_outer)
            p4 = geom.add_point([0.0, H, 0.0], mesh_size=lc_outer)
            
            l1 = geom.add_line(p1, p2)
            l2 = geom.add_line(p2, p3)
            l3 = geom.add_line(p3, p4)
            l4 = geom.add_line(p4, p1)
            outer_loop = geom.add_curve_loop([l1, l2, l3, l4])

            center = [L/2, H/2, 0.0]
            c0   = geom.add_point([center[0] + R, center[1], 0.0], mesh_size=lc_inner)
            c90  = geom.add_point([center[0], center[1] + R, 0.0], mesh_size=lc_inner)
            c180 = geom.add_point([center[0] - R, center[1], 0.0], mesh_size=lc_inner)
            c270 = geom.add_point([center[0], center[1] - R, 0.0], mesh_size=lc_inner)
            cen  = geom.add_point(center, mesh_size=lc_inner)

            a1 = geom.add_circle_arc(c0, cen, c90)
            a2 = geom.add_circle_arc(c90, cen, c180)
            a3 = geom.add_circle_arc(c180, cen, c270)
            a4 = geom.add_circle_arc(c270, cen, c0)
            hole_loop = geom.add_curve_loop([a1, a2, a3, a4])

            geom.add_plane_surface(outer_loop, holes=[hole_loop])
            mesh = geom.generate_mesh(order=2)

            tris6 = mesh.get_cells_type("triangle6")
            tris6_reordered = tris6[:, [0,1,2,4,5,3]]

        return mesh.points[:, :2], tris6_reordered

    # Initial guesses for mesh sizes
    lc_outer = 1.0
    lc_inner = 0.8

    tolerance = 0.05 * target_elements
    max_iterations = 10

    for _ in range(max_iterations):
        points, tris6_reordered = mesh_with_size(lc_outer, lc_inner)
        num_elements = tris6_reordered.shape[0]

        if abs(num_elements - target_elements) < tolerance:
            print(f"Converged to {num_elements} elements.")
            break

        # Adjust mesh sizes based on how close we are to the target
        scaling_factor = np.sqrt(num_elements / target_elements)
        lc_outer *= scaling_factor
        lc_inner *= scaling_factor

    return points, tris6_reordered


def generate_wall_with_hole_uniform_mesh(
    L=6.0,
    H=4.0,
    R=1.0,
    mesh_size=0.2,
    order=2
):
    """
    Rectangle [0,L]×[0,H] minus a circle of radius R at its center,
    meshed with 6-node (order=2) triangles, all elements ~mesh_size.
    """
    with pygmsh.geo.Geometry() as geom:
        # Define the four corner points of the rectangle, all with the same mesh_size
        p1 = geom.add_point([0.0, 0.0, 0.0], mesh_size=mesh_size)
        p2 = geom.add_point([  L, 0.0, 0.0], mesh_size=mesh_size)
        p3 = geom.add_point([  L,   H, 0.0], mesh_size=mesh_size)
        p4 = geom.add_point([0.0,   H, 0.0], mesh_size=mesh_size)
        l1 = geom.add_line(p1, p2)
        l2 = geom.add_line(p2, p3)
        l3 = geom.add_line(p3, p4)
        l4 = geom.add_line(p4, p1)
        outer_loop = geom.add_curve_loop([l1, l2, l3, l4])

        # Circle: again use the same mesh_size around the hole
        center = [L/2, H/2, 0.0]
        c0   = geom.add_point([center[0] + R, center[1],    0.0], mesh_size=mesh_size)
        c90  = geom.add_point([center[0],    center[1] + R, 0.0], mesh_size=mesh_size)
        c180 = geom.add_point([center[0] - R, center[1],    0.0], mesh_size=mesh_size)
        c270 = geom.add_point([center[0],    center[1] - R, 0.0], mesh_size=mesh_size)
        cen  = geom.add_point(center, mesh_size=mesh_size)
        a1 = geom.add_circle_arc(c0,   cen, c90)
        a2 = geom.add_circle_arc(c90,  cen, c180)
        a3 = geom.add_circle_arc(c180, cen, c270)
        a4 = geom.add_circle_arc(c270, cen, c0)
        hole_loop = geom.add_curve_loop([a1, a2, a3, a4])

        # Subtract hole
        geom.add_plane_surface(outer_loop, holes=[hole_loop])

        mesh = geom.generate_mesh(order=order)

    # extract nodes and 6-node triangles
    points = mesh.points[:, :2]
    tris6  = mesh.get_cells_type("triangle6")
    # reorder midside node indexing if necessary:
    tris6_reordered = tris6[:, [0,1,2,4,5,3]]

    return points, tris6_reordered

# =============================================================================
# generate + plot
# =============================================================================
# mesh   = generate_wall_with_hole()
# points = mesh.points[:, :2]
# tris6  = mesh.get_cells_type("triangle6")
# tris6_reordered = tris6[:,[0,1,2,4,5,3]]

# plt.figure(figsize=(10,6))
# for tri in tris6:
#     xy = points[tri]
#     # draw the three straight edges of each 6-node triangle
#     for i,j in [(0,1),(1,2),(2,0)]:
#         plt.plot(*xy[[i,j]].T, 'k-', lw=0.6)

# # overlay all nodes (including midside)
# plt.scatter(points[:,0], points[:,1], s=12, c='red', zorder=+5)

# plt.gca().set_aspect('equal', 'box')
# plt.xlabel("x"); plt.ylabel("y")
# plt.title("6-node LST mesh of a rectangle with a circular hole")
# plt.grid(True)
# plt.show()


