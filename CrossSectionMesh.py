import numpy as np
from scipy.sparse import coo_matrix

def point_in_polygon(p, poly):
    """射线法判断点 p 是否在多边形 poly 内部。"""
    p = np.asarray(p, dtype=float)
    poly = np.asarray(poly, dtype=float)
    M = len(poly)
    inside = False
    x, y = p
    for k in range(M):
        x1, y1 = poly[k]
        x2, y2 = poly[(k + 1) % M]
        if ((y1 > y) != (y2 > y)):
            xin = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < xin:
                inside = not inside
    return inside


def remove_elements_in_holes(nodes, elements, holes):
    """
    删除中心落在任何孔洞内部的单元。
    
    nodes:    (N, 2) 节点坐标
    elements: (E, 9) Q9 单元节点编号
    holes:    list of (M_k, 2) 孔洞多边形顶点
    返回：
        elements_kept: (E', 9) 剩余单元
    """
    keep = np.ones(len(elements), dtype=bool)
    for e in range(len(elements)):
        conn = elements[e]
        # 单元中心 = 第 9 个节点（Q9 中心节点）
        center = nodes[conn[8]]
        for hole in holes:
            if point_in_polygon(center, hole):
                keep[e] = False
                break
    return elements[keep]


# ============================================================
# 边界识别与分组
# ============================================================
def point_on_segment(p, A, B, tol=1e-8):
    AB = B - A
    AP = p - A
    AB_len2 = np.dot(AB, AB)
    if AB_len2 < 1e-30:
        return False
    cross = AB[0] * AP[1] - AB[1] * AP[0]
    if abs(cross) > tol * max(1.0, np.sqrt(AB_len2)):
        return False
    t = np.dot(AP, AB) / AB_len2
    return -tol <= t <= 1 + tol


def identify_boundary_nodes(nodes, all_polygons, tol=1e-8):
    nodes = np.asarray(nodes, dtype=float)
    N = len(nodes)
    is_boundary = np.zeros(N, dtype=bool)
    for poly in all_polygons:
        poly = np.asarray(poly, dtype=float)
        M = len(poly)
        for k in range(M):
            A = poly[k]
            B = poly[(k + 1) % M]
            for i in range(N):
                if is_boundary[i]:
                    continue
                if point_on_segment(nodes[i], A, B, tol):
                    is_boundary[i] = True
    return np.where(is_boundary)[0]


def classify_boundary_nodes(nodes, boundary_nodes, all_polygons, tol=1e-8):
    nodes = np.asarray(nodes, dtype=float)
    groups = []
    for poly in all_polygons:
        poly = np.asarray(poly, dtype=float)
        M = len(poly)
        poly_groups = [[] for _ in range(M)]
        for idx in boundary_nodes:
            p = nodes[idx]
            for k in range(M):
                A = poly[k]
                B = poly[(k + 1) % M]
                if point_on_segment(p, A, B, tol):
                    poly_groups[k].append(idx)
                    break
        groups.append(poly_groups)
    return groups


# ============================================================
# 仿射推断
# ============================================================
def compute_boundary_new_affine(nodes, boundary_nodes, polygons_old, polygons_new):
    nodes = np.asarray(nodes, dtype=float)
    boundary_nodes = np.asarray(boundary_nodes, dtype=int)
    groups = classify_boundary_nodes(nodes, boundary_nodes, polygons_old)

    node_new = {}
    for pid, poly_old in enumerate(polygons_old):
        poly_old = np.asarray(poly_old, dtype=float)
        poly_new = np.asarray(polygons_new[pid], dtype=float)
        M = len(poly_old)
        for k in range(M):
            A_old = poly_old[k]
            B_old = poly_old[(k + 1) % M]
            A_new = poly_new[k]
            B_new = poly_new[(k + 1) % M]
            AB_old = B_old - A_old
            AB_new = B_new - A_new
            AB_old_len2 = np.dot(AB_old, AB_old)
            if AB_old_len2 < 1e-30:
                continue
            for idx in groups[pid][k]:
                p = nodes[idx]
                t = np.dot(p - A_old, AB_old) / AB_old_len2
                node_new[idx] = A_new + t * AB_new

    return np.array([node_new[idx] for idx in boundary_nodes])


# ============================================================
# 拉普拉斯变形
# ============================================================
def deform_mesh_laplacian(nodes, elements, boundary_nodes, boundary_new):
    nodes = np.asarray(nodes, dtype=float)
    elements = np.asarray(elements, dtype=int)
    boundary_nodes = np.asarray(boundary_nodes, dtype=int).ravel()
    boundary_new = np.asarray(boundary_new, dtype=float)

    N = nodes.shape[0]
    E = elements.shape[0]
    interior_nodes = np.setdiff1d(np.arange(N), boundary_nodes)

    rows, cols = [], []
    for e in range(E):
        conn = elements[e]
        for i in range(9):
            for j in range(i + 1, 9):
                rows.append(conn[i]); cols.append(conn[j])
                rows.append(conn[j]); cols.append(conn[i])

    rows = np.array(rows)
    cols = np.array(cols)
    vals = -np.ones(len(rows))

    L = coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()
    deg = -np.array(L.sum(axis=1)).ravel()
    L = L + coo_matrix((deg, (np.arange(N), np.arange(N))), shape=(N, N)).tocsr()

    X_new = np.zeros((N, 2))
    X_new[boundary_nodes, :] = boundary_new

    L_ii = L[interior_nodes, :][:, interior_nodes]
    L_ib = L[interior_nodes, :][:, boundary_nodes]

    X_new[interior_nodes, :] = -np.linalg.solve(L_ii.toarray(), L_ib @ boundary_new)

    return X_new

def project_centers_and_midpoints(nodes, elements, polygons, tol=1e-8):
    nodes = nodes.copy()
    E = elements.shape[0]

    edge_defs = [
        (0, 1, 4), (1, 2, 5), (2, 3, 6), (3, 0, 7),
    ]

    # 预先把所有多边形转成 (M, 2) 数组
    polys = []
    for poly in polygons:
        arr = np.asarray(poly, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2:
            continue   # 跳过无效多边形
        polys.append(arr)

    def is_on_polygon(p):
        for poly in polys:
            M = len(poly)
            for k in range(M):
                A = poly[k]
                B = poly[(k + 1) % M]
                AB = B - A
                AP = p - A
                AB_len2 = np.dot(AB, AB)
                if AB_len2 < 1e-30:
                    continue
                cross = AB[0] * AP[1] - AB[1] * AP[0]
                if abs(cross) > tol * max(1.0, np.sqrt(AB_len2)):
                    continue
                t = np.dot(AP, AB) / AB_len2
                if -tol <= t <= 1 + tol:
                    return True
        return False

    # ---------- 1. 边中点投影 ----------
    edge_map = {}
    for e in range(E):
        conn = elements[e]
        for (ia, ib, im) in edge_defs:
            A = conn[ia]
            B = conn[ib]
            Mnode = conn[im]
            key = (min(A, B), max(A, B))
            if key not in edge_map:
                edge_map[key] = (A, B, Mnode)

    for (A, B, Mnode) in edge_map.values():
        if is_on_polygon(nodes[Mnode]):
            continue
        nodes[Mnode] = 0.5 * (nodes[A] + nodes[B])

    # ---------- 2. 第 9 个节点设为四角平均 ----------
    for e in range(E):
        conn = elements[e]
        A, B, C, D = conn[0], conn[1], conn[2], conn[3]
        M9 = conn[8]
        nodes[M9] = 0.25 * (nodes[A] + nodes[B] + nodes[C] + nodes[D])

    return nodes


# ============================================================
# 主流程
# ============================================================
def deform_mesh_auto(nodes, elements, outer_old, outer_new, holes_old=None, holes_new=None):
    """
    返回：
        nodes_new:      (N', 2) 变形后节点坐标(重编号后)
        boundary_nodes: (B,)   边界节点编号(重编号后,0-based)
        boundary_new:   (B, 2) 边界节点新坐标
        elements:       (E', 9) 保留单元（重编号后）
        nodes_old:      (N', 2) 原始节点坐标(重编号后)
    """
    nodes = np.asarray(nodes, dtype=float)
    elements = np.asarray(elements, dtype=int) - 1 # 1-based for Mathematica

    if holes_old is None:
        holes_old = []
    if holes_new is None:
        holes_new = []

    polygons_old = [np.asarray(outer_old, dtype=float)] + [np.asarray(h, dtype=float) for h in holes_old]
    polygons_new = [np.asarray(outer_new, dtype=float)] + [np.asarray(h, dtype=float) for h in holes_new]

    if len(polygons_old) != len(polygons_new):
        raise ValueError("内部边界多边形数量不一致")

    # 1. 删除孔洞内部的单元
    if len(holes_old) > 0:
        elements = remove_elements_in_holes(nodes, elements, holes_old)

    # 2. 删除孤立节点，重新编号
    used_nodes = np.unique(elements)
    remap = -np.ones(nodes.shape[0], dtype=int)
    remap[used_nodes] = np.arange(len(used_nodes))

    nodes_old = nodes[used_nodes]
    elements = remap[elements]

    # 3. 识别所有边界节点
    boundary_nodes = identify_boundary_nodes(nodes_old, polygons_old)

    # 4. 仿射推断所有边界节点新坐标
    boundary_new = compute_boundary_new_affine(nodes_old, boundary_nodes, polygons_old, polygons_new)

    # 5. 拉普拉斯变形
    nodes_new = deform_mesh_laplacian(nodes_old, elements, boundary_nodes, boundary_new)

    # 6. 后处理：边中点和中心节点投影
    nodes_new = project_centers_and_midpoints(nodes_new, elements, boundary_nodes)

    return nodes_new.tolist(), boundary_nodes.tolist(), boundary_new.tolist(), elements.tolist(), nodes_old.tolist()
