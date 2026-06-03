#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <random>
using namespace std;
using ll = long long;

struct Edge {
    int u, v;
    ll w;
    bool operator<(const Edge& other) const {
        return w < other.w;
    }
};

// Система непересекающихся множеств (DSU)
class DSU {
    vector<int> parent, rank;
public:
    DSU(int n) {
        parent.resize(n);
        rank.resize(n, 0);
        for (int i = 0; i < n; ++i) parent[i] = i;
    }
    int find(int x) {
        if (parent[x] != x)
            parent[x] = find(parent[x]);
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

// Генерация случайного неориентированного графа (список рёбер)
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

pair<ll, vector<Edge>> kruskal(int n, vector<Edge> edges) {
    sort(edges.begin(), edges.end());
    DSU dsu(n);
    vector<Edge> mst;
    ll total_weight = 0;
    for (const Edge& e : edges) {
        if (dsu.unite(e.u, e.v)) {
            mst.push_back(e);
            total_weight += e.w;
            if ((int)mst.size() == n - 1) break;
        }
    }
    return {total_weight, mst};
}

int main(int argc, char* argv[]) {
    int n = 100;
    if (argc > 1) n = atoi(argv[1]);
    double density = 0.2;
    vector<Edge> edges = generate_edges(n, density, 123);

    auto start = chrono::high_resolution_clock::now();
    auto result = kruskal(n, edges);
    auto end = chrono::high_resolution_clock::now();
    double elapsed = chrono::duration<double>(end - start).count();

    cout << "Sequential Kruskal, n=" << n << ", time=" << elapsed << " s, MST weight=" << result.first << endl;
    return 0;
}