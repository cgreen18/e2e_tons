/*
--------------------------------------------------------------------------------
Copyright (c) 2025 Purdue University
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met: redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer;
redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution;
neither the name of the copyright holders nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Author(s): Conor Green

--------------------------------------------------------------------------------
Description: TODO
*/

#pragma once

#include <vector>
#include <iostream>
#include <chrono>
#include <cstring>
#include <fstream>

// using namespace std;
using std::vector;
using std::string;

using std::cout;
using std::cerr;
using std::cin;
using std::endl;
using std::flush;
using std::getline;


// outputs and prints
////////////////////////////////////////////////////////////////////////////////

void print_twod_vector(const vector<vector<int>>& dists) {
    int n = dists.size();

    // optional: header row with destination node indices
    cout << "    ";
    for (int j = 0; j < n; j++) {
        cout << j << " ";
    }
    cout << "\n";

    // separator line
    cout << "    ";
    for (int j = 0; j < n; j++) {
        cout << "--";
    }
    cout << "\n";

    // each row: source node i, then distances to all j
    for (int i = 0; i < n; i++) {
        cout << i << " | ";
        for (int j = 0; j < n; j++) {
            cout << dists[i][j] << " ";
        }
        cout << "\n";
    }
}

struct Timer {
    std::chrono::high_resolution_clock::time_point start;

    Timer() {
        reset();
    }

    void reset() {
        start = std::chrono::high_resolution_clock::now();
    }

    // returns elapsed milliseconds as double
    double ms() const {
        auto now = std::chrono::high_resolution_clock::now();
        auto dur = std::chrono::duration_cast<std::chrono::microseconds>(now - start).count();
        // convert microseconds to milliseconds with fractional part
        return dur / 1000.0;
    }

    // returns elapsed seconds as double
    double s() const {
        auto now = std::chrono::high_resolution_clock::now();
        auto dur = std::chrono::duration_cast<std::chrono::microseconds>(now - start).count();
        // convert microseconds to milliseconds with fractional part
        return dur / 1000000.0;
    }

    void print(const string& label) const {
        double elapsed_ms = ms();
        double elapsed_s  = elapsed_ms / 1000.0;
        double elapsed_m  = elapsed_s  / 60.0;

        cout << label << "Elapsed time            :"
                  << elapsed_ms << " ms, "
                  << elapsed_s  << " s, "
                  << elapsed_m  << " min\n";
    }
};

void print_status(const int n_conns, const int n_remaining, const int n_possible, const double avg_hops, Timer& timer, const string& label){

    cout << "----------------------------------------" << endl;
    cout << "STATUS :: " << label << endl;
    cout << "   connected               :" << n_conns << " / " << n_conns + n_remaining << " (unidirectionally, " << n_conns/2 << "/" << (n_conns + n_remaining)/2 << ")" << endl;
    cout << "   remaining               :" << n_remaining << " (unidirectionally, " << n_remaining/2 << ")" << endl;
    cout << "   possible connections    :" << 2*n_possible << " (unidirectionally, " << n_possible << ")" << endl;
    cout << "   avg hops                :" << avg_hops << endl;
    timer.print("   ");
    cout << "----------------------------------------" << endl;
}

void wait_for_enter(const char* prompt = "Press Enter to continue...")
{
    cout << prompt << flush;
    string dummy;
    getline(cin, dummy);   // waits until user hits Enter
}

void write_adj_matrix(const vector<vector<int>>& adj,const string& path){
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Failed to open file: " + path);

    const int n = static_cast<int>(adj.size());
    for (int i = 0; i < n; ++i) {
        const auto& row = adj[i];
        for (int j = 0; j < static_cast<int>(row.size()); ++j) {
            if (j) out << ' ';          // space before all but first
            out << row[j];
        }
        out << '\n';                     // line break per row
    }
    // ofstream closes on destruction
}

void log_metrics(const string& file_path, int iter, double elapsed_ms, const double avg_hops, const int n_links)
{
    static bool first_call = true;

    // open mode: truncate on first call, append thereafter
    std::ios_base::openmode mode = std::ios::out | (first_call ? std::ios::trunc : std::ios::app);
    std::ofstream out(file_path, mode);
    if (!out) {
        cerr << "ERROR: cannot open log file: " << file_path << "\n";
        return;
    }

    if (first_call) {
        // out << "iter,elapsed_ms,seconds,minutes,n_links,avg_hops\n";
        out << "iter,seconds,minutes,n_links,avg_hops\n";
        first_call = false;
    }

    double seconds = elapsed_ms / 1000.0;
    double minutes = seconds / 60.0;

    out << iter << ','
        // << elapsed_ms << ','
        << seconds << ','
        << minutes << ','
        << n_links << ','
        << avg_hops
        << '\n';
}