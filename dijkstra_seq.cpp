#include <iostream>
#include <vector>
#include <queue>
#include <chrono>
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

vector<ll> dijkstra(int n, const vector<vector<pair<int,ll>>>& g, int src) {
    vector<ll> dist(n, INF);
    dist[src] = 0;
    priority_queue<pair<ll,int>, vector<pair<ll,int>>, greater<>> pq;
    pq.push({0, src});

    while (!pq.empty()) {
        ll d = pq.top().first;
        int u = pq.top().second;
        pq.pop();
        if (d != dist[u]) continue;
        for (size_t i = 0; i < g[u].size(); ++i) {
            int v = g[u][i].first;
            ll w = g[u][i].second;
            if (dist[v] > d + w) {
                dist[v] = d + w;
                pq.push({dist[v], v});
            }
        }
    }
    return dist;
}

int main(int argc, char* argv[]) {
    int n = 1000;
    if (argc > 1) n = atoi(argv[1]);
    double density = 0.2;
    int src = 0;

    auto g = generate_graph(n, density);
    auto start = chrono::high_resolution_clock::now();
    auto dist = dijkstra(n, g, src);
    auto end = chrono::high_resolution_clock::now();
    double elapsed = chrono::duration<double>(end - start).count();

    cout << "Sequential Dijkstra, n=" << n << ", time=" << elapsed << " s" << endl;

    return 0;
}