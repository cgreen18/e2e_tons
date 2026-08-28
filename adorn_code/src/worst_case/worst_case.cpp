#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <limits>
#include <iomanip>
#include <filesystem>

using namespace std;

// Solves the assignment problem (Hungarian algorithm) in O(N^3) time.
// It finds the minimum cost perfect matching in a bipartite graph.
double hungarian_min_cost(int n, const vector<vector<double>>& cost) {
    vector<double> u(n + 1, 0), v(n + 1, 0), minv(n + 1);
    vector<int> p(n + 1, 0), way(n + 1, 0);
    
    for (int i = 1; i <= n; ++i) {
        p[0] = i;
        int j0 = 0;
        minv.assign(n + 1, numeric_limits<double>::infinity());
        vector<char> used(n + 1, false);
        
        do {
            used[j0] = true;
            int i0 = p[j0], j1 = 0;
            double delta = numeric_limits<double>::infinity();
            
            for (int j = 1; j <= n; ++j) {
                if (!used[j]) {
                    double cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
                    if (cur < minv[j]) {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if (minv[j] < delta) {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            for (int j = 0; j <= n; ++j) {
                if (used[j]) {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
        } while (p[j0] != 0);
        
        do {
            int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0 != 0);
    }
    
    // Sum the minimum costs of the matching
    double total_cost = 0;
    for (int j = 1; j <= n; ++j) {
        if (p[j] != 0) {
            total_cost += cost[p[j] - 1][j - 1];
        }
    }
    return total_cost;
}

// Loads a space-delimited text file into an adjacency matrix.
vector<vector<double>> load_adjacency_matrix(const string& file_path) {
    vector<vector<double>> matrix;
    ifstream file(file_path);
    if (!file.is_open()) {
        cerr << "Error: Could not open file " << file_path << endl;
        exit(1);
    }

    string line;
    while (getline(file, line)) {
        if (line.empty()) continue;
        vector<double> row;
        stringstream ss(line);
        double val;
        while (ss >> val) {
            row.push_back(val);
        }
        matrix.push_back(row);
    }
    return matrix;
}

string extract_topo_name(const string& file_path) {
    return filesystem::path(file_path).stem().string();
}

string format_bottleneck_link(const pair<int, int>& channel) {
    return to_string(channel.first) + "->" + to_string(channel.second);
}

void write_worst_case_metrics(const string& topo, double worst_thru,
                              const pair<int, int>& bottleneck) {
    const string metrics_dir = "files/metrics";
    filesystem::create_directories(metrics_dir);

    ostringstream row;
    row << fixed << setprecision(4) << topo << "," << worst_thru << ","
        << format_bottleneck_link(bottleneck) << "\n";
    const string line = row.str();

    const string aggregate_path = metrics_dir + "/worst_case.csv";
    bool write_header = !filesystem::exists(aggregate_path) ||
                        filesystem::file_size(aggregate_path) == 0;
    ofstream aggregate(aggregate_path, ios::app);
    if (!aggregate.is_open()) {
        cerr << "Error: Could not open " << aggregate_path << " for writing" << endl;
        return;
    }
    if (write_header) {
        aggregate << "topo,worst_thru,bottleneck_link\n";
    }
    aggregate << line;

    const string topo_path = metrics_dir + "/worst_case_" + topo + ".csv";
    ofstream topo_file(topo_path, ios::trunc);
    if (!topo_file.is_open()) {
        cerr << "Error: Could not open " << topo_path << " for writing" << endl;
        return;
    }
    topo_file << line;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: ./worst_case_throughput <adjacency_matrix.map>" << endl;
        return 1;
    }

    string file_path = argv[1];
    vector<vector<double>> adj_matrix = load_adjacency_matrix(file_path);
    int n = adj_matrix.size();
    
    // Identify all channels with bandwidth > 0
    vector<pair<int, int>> channels;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (adj_matrix[i][j] > 0) {
                channels.push_back({i, j});
            }
        }
    }

    // ---------------------------------------------------------
    // 1. Define Oblivious Routing Function (pi)
    // ---------------------------------------------------------
    // Construct deterministic shortest-path route (Floyd-Warshall)
    vector<vector<double>> dist(n, vector<double>(n, numeric_limits<double>::infinity()));
    vector<vector<int>> next_hop(n, vector<int>(n, -1));

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) {
                dist[i][j] = 0;
            } else if (adj_matrix[i][j] > 0) {
                dist[i][j] = 1.0; 
                next_hop[i][j] = j;
            }
        }
    }

    for (int k = 0; k < n; ++k) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (dist[i][k] + dist[k][j] < dist[i][j]) {
                    dist[i][j] = dist[i][k] + dist[k][j];
                    next_hop[i][j] = next_hop[i][k];
                }
            }
        }
    }

    // ---------------------------------------------------------
    // 2. Bipartite Graph Construction & Max-Weight Matching
    // ---------------------------------------------------------
    double worst_case_ideal_throughput = numeric_limits<double>::infinity();
    pair<int, int> bottleneck_channel = {-1, -1};

    cout << "Calculating worst-case throughput..." << endl;

    for (const auto& c : channels) {
        int u = c.first;
        int v = c.second;
        double b_c = adj_matrix[u][v];

        // Build the cost matrix for this channel
        // Note: To find max-weight matching using a min-cost solver, we negate the weights
        vector<vector<double>> cost_matrix(n, vector<double>(n, 0.0));

        for (int s = 0; s < n; ++s) {
            for (int d = 0; d < n; ++d) {
                if (s == d) continue;

                int curr = s;
                bool uses_channel = false;
                int hops = 0;

                while (curr != d && curr != -1 && hops < n) {
                    int nxt = next_hop[curr][d];
                    if (curr == u && nxt == v) {
                        uses_channel = true;
                        break;
                    }
                    curr = nxt;
                    hops++;
                }

                if (uses_channel) {
                    cost_matrix[s][d] = -1.0; // Negated load for the min-cost solver
                }
            }
        }

        // Solve assignment problem to find gamma_{c,max}
        double min_cost = hungarian_min_cost(n, cost_matrix);
        double gamma_c_max = -min_cost;

        // ---------------------------------------------------------
        // 3. Calculate Ideal Worst-Case Throughput Bound
        // ---------------------------------------------------------
        if (gamma_c_max > 0) {
            double channel_throughput = b_c / gamma_c_max;
            if (channel_throughput < worst_case_ideal_throughput) {
                worst_case_ideal_throughput = channel_throughput;
                bottleneck_channel = c;
            }
        }
    }

    if (bottleneck_channel.first != -1) {
        cout << fixed << setprecision(4);
        cout << "Worst-case ideal throughput: " << worst_case_ideal_throughput << endl;
        cout << "Bottleneck Channel (source, dest): (" 
             << bottleneck_channel.first << ", " << bottleneck_channel.second << ")" << endl;
        write_worst_case_metrics(extract_topo_name(file_path),
                                 worst_case_ideal_throughput,
                                 bottleneck_channel);
    } else {
        cout << "No traffic loads found; throughput bound is infinite." << endl;
    }



    return 0;
}
