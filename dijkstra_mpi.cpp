#include <mpi.h>
#include <iostream>
#include <vector>
#include <queue>
#include <random>
#include <algorithm>
using namespace std;
using ll = long long;

const ll INF = 1e18;

vector<vector<pair<int, ll>>> generate_graph(int n, double density = 0.2, int seed = 123) {
    mt19937 rng(seed);
    uniform_int_distribution<int> weight_dist(1, 100);
    uniform_real_distribution<double> edge_dist(0.0, 1.0);
    vector<vector<pair<int, ll>>> g(n);
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (edge_dist(rng) < density) {
                ll w = weight_dist(rng);
                g[i].emplace_back(j, w);
                g[j].emplace_back(i, w);
            }
        }
    }
    return g;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int n = 100;
    if (argc > 1) n = atoi(argv[1]);
    auto g = generate_graph(n);

    int local_n = n / size;
    int rem = n % size;
    int start = rank * local_n + min(rank, rem);
    int end = start + local_n + (rank < rem ? 1 : 0);
    int local_count = end - start;

    vector<ll> dist(n, INF);
    vector<bool> visited(n, false);
    dist[0] = 0;

    using P = pair<ll, int>;
    priority_queue<P, vector<P>, greater<P>> pq;
    if (0 >= start && 0 < end) pq.push({0, 0});

    double t_start = MPI_Wtime();
    int processed = 0;

    while (processed < n) {
        ll local_min_val = INF;
        int local_min_global_idx = -1;

        while (!pq.empty()) {
            auto [d, u] = pq.top();
            if (d == dist[u] && !visited[u]) {
                local_min_val = d;
                local_min_global_idx = u;
                break;
            }
            pq.pop();
        }

        struct { ll val; int idx; } local = {local_min_val, local_min_global_idx};
        vector<decltype(local)> all_mins;
        if (rank == 0) all_mins.resize(size);
        MPI_Gather(&local, sizeof(local), MPI_BYTE,
                   all_mins.data(), sizeof(local), MPI_BYTE,
                   0, MPI_COMM_WORLD);

        int global_rank = -1, global_idx = -1;
        ll global_val = INF;
        if (rank == 0) {
            for (int p = 0; p < size; ++p) {
                if (all_mins[p].val < global_val) {
                    global_val = all_mins[p].val;
                    global_idx = all_mins[p].idx;
                    global_rank = p;
                }
            }
        }
        MPI_Bcast(&global_rank, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&global_idx, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&global_val, 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);

        if (global_idx == -1) break;

        if (rank == global_rank) {
            visited[global_idx] = true;
            while (!pq.empty() && pq.top().second == global_idx) pq.pop();
        }

        for (const auto& edge : g[global_idx]) {
            int v = edge.first;
            ll w = edge.second;
            if (visited[v]) continue;
            if (dist[v] > global_val + w) {
                dist[v] = global_val + w;
                if (v >= start && v < end) pq.push({dist[v], v});
            }
        }
        processed++;
    }

    double t_end = MPI_Wtime();
    if (rank == 0) {
        double elapsed = t_end - t_start;
        cout << "MPI Dijkstra (heap), n=" << n << ", p=" << size << ", time=" << elapsed << " s" << endl;
    }
    MPI_Finalize();
    return 0;
}