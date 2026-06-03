#include <mpi.h>
#include <windows.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
using namespace std;
using ll = long long;

struct Edge {
    int u, v;
    ll w;
    bool operator<(const Edge& other) const { return w < other.w; }
};

class DSU {
    vector<int> parent, rank;
public:
    DSU(int n) {
        parent.resize(n);
        rank.resize(n, 0);
        for (int i = 0; i < n; ++i) parent[i] = i;
    }
    int find(int x) {
        if (parent[x] != x) parent[x] = find(parent[x]);
        return parent[x];
    }
    bool unite(int x, int y) {
        x = find(x); y = find(y);
        if (x == y) return false;
        if (rank[x] < rank[y]) swap(x, y);
        parent[y] = x;
        if (rank[x] == rank[y]) rank[x]++;
        return true;
    }
};

vector<Edge> generate_edges(int n, double density = 0.2, int seed = 123) {
    mt19937 rng(seed);
    uniform_int_distribution<ll> weight_dist(1, 100);
    uniform_real_distribution<double> edge_dist(0.0, 1.0);
    vector<Edge> edges;
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (edge_dist(rng) < density) {
                edges.push_back({i, j, weight_dist(rng)});
            }
        }
    }
    return edges;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = 100;
    if (argc > 1) n = atoi(argv[1]);

    vector<Edge> all_edges;
    int total_edges = 0;

    double t_start = 0.0;
    if (rank == 0) {
        t_start = MPI_Wtime();
        all_edges = generate_edges(n);
        sort(all_edges.begin(), all_edges.end());
        total_edges = all_edges.size();
    }
    MPI_Bcast(&total_edges, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (total_edges == 0) {
        if (rank == 0) cout << "No edges" << endl;
        MPI_Finalize();
        return 0;
    }

    int per_proc = total_edges / size;
    int rem = total_edges % size;
    int start_idx = rank * per_proc + min(rank, rem);
    int end_idx = start_idx + per_proc + (rank < rem ? 1 : 0);
    int local_cnt = end_idx - start_idx;

    vector<Edge> local_edges(local_cnt);
    if (rank == 0) {
        int offset = 0;
        for (int p = 0; p < size; ++p) {
            int cnt = per_proc + (p < rem ? 1 : 0);
            if (p == 0) {
                local_edges.assign(all_edges.begin() + offset, all_edges.begin() + offset + cnt);
            } else {
                MPI_Send(&all_edges[offset], cnt * sizeof(Edge), MPI_BYTE, p, 0, MPI_COMM_WORLD);
            }
            offset += cnt;
        }
    } else {
        MPI_Recv(local_edges.data(), local_cnt * sizeof(Edge), MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    DSU dsu(n);
    vector<Edge> local_mst;
    for (const Edge& e : local_edges) {
        if (dsu.unite(e.u, e.v)) local_mst.push_back(e);
    }

    int local_mst_size = local_mst.size();
    if (rank == 0) {
        vector<Edge> all_mst = local_mst;
        for (int p = 1; p < size; ++p) {
            int cnt;
            MPI_Recv(&cnt, 1, MPI_INT, p, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (cnt > 0) {
                vector<Edge> tmp(cnt);
                MPI_Recv(tmp.data(), cnt * sizeof(Edge), MPI_BYTE, p, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                all_mst.insert(all_mst.end(), tmp.begin(), tmp.end());
            }
        }
        sort(all_mst.begin(), all_mst.end());
        DSU dsu_final(n);
        ll total_weight = 0;
        int added = 0;
        for (const Edge& e : all_mst) {
            if (dsu_final.unite(e.u, e.v)) {
                total_weight += e.w;
                added++;
                if (added == n - 1) break;
            }
        }
        double t_end = MPI_Wtime();
        double elapsed = t_end - t_start;
        cout << "MPI Kruskal, n=" << n << ", p=" << size << ", time=" << elapsed << " s, MST weight=" << total_weight << endl;
    } else {
        MPI_Send(&local_mst_size, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
        if (local_mst_size > 0)
            MPI_Send(local_mst.data(), local_mst_size * sizeof(Edge), MPI_BYTE, 0, 2, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}