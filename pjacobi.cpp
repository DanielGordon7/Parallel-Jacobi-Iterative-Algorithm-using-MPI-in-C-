#include <mpi.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <iomanip>
#include <string>
#include <sstream>
#include <stdlib.h>
#include <math.h>
#include <cmath>
#include <algorithm>

#define NDIM 2
#define MAX_ITR 1000000

using namespace std;
bool DEBUG = false;


int main(int argc, char* argv[]) {

    double starttime, endtime;
    MPI_Init(&argc, &argv);

    int wrank, crank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &wrank);
    MPI_Comm cart_comm;

    double q = sqrt(size);
    int dims[NDIM] = {int(q), int(q)};
    int period[NDIM] = {0, 0};
    int coords[NDIM];
    if (DEBUG)
        if (wrank == 0) printf("grid dims: (%d x %d) \n", dims[0], dims[1]);
    MPI_Cart_create(MPI_COMM_WORLD, NDIM, dims, period, 1, &cart_comm);
    MPI_Comm_rank(cart_comm, &crank);
    MPI_Cart_coords(cart_comm, crank, NDIM, coords);

    int N;
    string out_name = argv[3];

    // read matrix A data
    if (crank == 0) {
        string matrix_file = argv[1];
        ifstream matrix_data;
        matrix_data.open(matrix_file);
        if(!matrix_data) {
            cout << "couldn't open matrix file " << matrix_file << endl;
            return MPI_Finalize();
            exit(1);
        }
        string line;
        getline(matrix_data, line);
        N = stoi(line);
        matrix_data.close();
    }
    MPI_Bcast(&N, 1, MPI_INT, 0, cart_comm);
    int threshold = N % (int)q;

    double A[N][N] = {};
    vector<double> d_vec(N, 0.0);

    if (crank == 0) {
        string matrix_file = argv[1];
        ifstream matrix_data;
        matrix_data.open(matrix_file);
        matrix_data.ignore(256, '\n');
        string matrix_line;
        for (int row=0; row<N; row++) {
            double value;
            int col = 0;
            getline(matrix_data, matrix_line);
            stringstream linestream(matrix_line);
            while (linestream >> value) {
                A[row][col] = value;
                col++;
            }
        }
        matrix_data.close();
        // store diagonal elements
        for (int i=0; i<N; i++)
            d_vec[i] = A[i][i];
        
        if (DEBUG) {
            printf("N: %i \n", N);
            cout << "Matrix A elements: " << endl;
            for (int i=0; i<N; i++) {
                for (int j=0; j<N; j++) cout << A[i][j] << " ";
                cout << endl;
            }
        }
    }
    // read vector b data
    vector<double> b_vec(N, 0.0);
    if (crank == 0) {
        string vec_file = argv[2];
        ifstream vec_data;
        vec_data.open(vec_file);
        if(!vec_data) {
            cout << "couldn't open vector file " << vec_file << endl;
            return MPI_Finalize();
            exit(1);
        }
        double value;
        string line;
        int i = 0;
        getline(vec_data, line);
        stringstream linestream(line);
        while (linestream >> value) {
            b_vec[i] = value;
            i++;
        }
        vec_data.close();
        if (DEBUG) {
            cout << "Vector b elements: ";
            for (int i=0; i<N; i++) cout << b_vec[i] << " ";
            cout << endl;
            cout << "-----------------------------------------" << endl;
        }
    }

    // create row & col communicators
    MPI_Comm col_comm, row_comm;
    MPI_Comm_split(cart_comm, coords[0], crank, &col_comm);
    MPI_Comm_split(cart_comm, coords[1], crank, &row_comm);
    int colrank, rowrank;
    MPI_Comm_rank(col_comm, &colrank);
    MPI_Comm_rank(row_comm, &rowrank);

    // scatter diagonal elements of A and vector b in col1 comm
    int prefix_sum = 0;
    vector<int> vec_scounts(q), vec_displs(q);
    for (int i = 0; i < q; i++) {
        vec_scounts[i] = (i < threshold) ? ceil(N/q) : floor(N/q);
        vec_displs[i] = prefix_sum;
        prefix_sum += vec_scounts[i];
    }
    if (DEBUG) {
        if (crank == 0) {
            printf("crank %i, vector scatter\n", crank);
            cout << "vec_scounts: ";
            for (int i = 0; i < q; i++) cout << vec_scounts[i] << " ";
            cout << endl;
            cout << "vec_displs: ";
            for (int i = 0; i < q; i++) cout << vec_displs[i] << " ";
            cout << endl;
        }
    } 

    vector<double> stored_d, stored_b;
    if (coords[0] == 0) {
        stored_d.resize(ceil(N/q), 0.0);
        stored_b.resize(ceil(N/q), 0.0);
        MPI_Scatterv(d_vec.data(), vec_scounts.data(), vec_displs.data(), MPI_DOUBLE, stored_d.data(), ceil(N/q), MPI_DOUBLE, 0, col_comm);
        MPI_Scatterv(b_vec.data(), vec_scounts.data(), vec_displs.data(), MPI_DOUBLE, stored_b.data(), ceil(N/q), MPI_DOUBLE, 0, col_comm);
        if (coords[1] >= threshold && threshold != 0) {
            stored_d.pop_back();
            stored_b.pop_back();
        }
        if (DEBUG) {
            printf("crank %i, stored_d: ", crank);
            for (int i = 0; i < stored_d.size(); i++) cout << stored_d[i] << " ";
            cout << endl;
            printf("crank %i, stored_b: ", crank);
            for (int i = 0; i < stored_b.size(); i++) cout << stored_b[i] << " ";
            cout << endl;
        } 
    }


    // scatter cols of A to processors in row1
    MPI_Datatype tmp, col_type;
    MPI_Type_vector(N, ceil(N/q), N, MPI_DOUBLE, &tmp);
    MPI_Type_create_resized(tmp, 0, ceil(N/q)*sizeof(double), &col_type);
    MPI_Type_commit(&col_type);

    vector<double> col_data;
    if (coords[1] == 0) {
        vector<int> row_scounts(ceil(N/q));
        vector<int> row_displs(ceil(N/q));
        for (int i = 0; i < q; i++) {
            row_scounts[i] = 1;
            row_displs[i] = i;
        }

        MPI_Scatterv(A, row_scounts.data(), row_displs.data(), col_type, A, ceil(N/q), col_type, 0, row_comm);

        if (DEBUG) {
            if (coords[0] == 0) {
                printf("crank %i, row1 broadcast\n", crank);
                cout << "scounts: ";
                for (int i=0; i<q; i++) cout << row_scounts[i] << " ";  // 1 1
                cout << endl << "displs: ";
                for (int i=0; i<q; i++) cout << row_displs[i] << " ";  // 0 1
                cout << endl;
            }
        }
        // move cols of A into 1D array
        if (coords[0] < threshold) {
            col_data.resize(N*ceil(N/q));
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < ceil(N/q); j++)
                    col_data[i*ceil(N/q) + j] = A[i][j];
            }
        } else if (coords[0] >= threshold) {
            col_data.resize(N*floor(N/q));
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < floor(N/q); j++)
                    col_data[i*floor(N/q) + j] = A[i][j];
            }
        }
    }

    // scatter cols of A down each col comm
    vector<int> col_scounts(q), col_displs(q);
    int prefix = 0;
    for (int i = 0; i < q; i++) {
        int num_rows, num_cols;
        num_cols = (coords[0] < threshold) ? ceil(N/q) : floor(N/q);
        num_rows = (i < threshold) ? ceil(N/q) : floor(N/q);
        col_scounts[i] = num_rows * num_cols;
        col_displs[i] = prefix;
        prefix += col_scounts[i];
    }

    if (DEBUG) {
        if (coords[1] == 0) {
            printf("crank %i, column broadcast\n", crank);
            cout << "scounts: ";
            for (int i = 0; i < q; i++) cout << col_scounts[i] << " ";
            cout << endl;
            cout << "displs: ";
            for (int i = 0; i < q; i++) cout << col_displs[i] << " ";
            cout << endl;
        }
    }

    int max_msg = ceil(N/q)*ceil(N/q);
    vector<double> proc_data(max_msg);
    MPI_Scatterv(col_data.data(), col_scounts.data(), col_displs.data(), MPI_DOUBLE, proc_data.data(), max_msg, MPI_DOUBLE, 0, col_comm);

    // store submatrix of A into 2D vector
    vector<vector<double>> local_A;
    int x = coords[0];
    int y = coords[1];
    if (x < threshold && y < threshold) {
        local_A.resize(ceil(N/q), vector<double> (ceil(N/q)));
        for (int i=0; i<local_A.size(); i++) {
            for (int j=0; j<local_A[0].size(); j++)
                local_A[i][j] = proc_data[i*ceil(N/q)+j];
        }
    } 
    else if (x < threshold && y >= threshold) {
        local_A.resize(floor(N/q), vector<double> (ceil(N/q)));
        for (int i=0; i<local_A.size(); i++) {
            for (int j=0; j<local_A[0].size(); j++)
                local_A[i][j] = proc_data[i*ceil(N/q)+j];
        }
    }
    else if (x >= threshold && y < threshold) {
        local_A.resize(ceil(N/q), vector<double> (floor(N/q)));
        for (int i=0; i<local_A.size(); i++) {
            for (int j=0; j<local_A[0].size(); j++) 
                local_A[i][j] = proc_data[i*floor(N/q)+j];
        }
    }
    else if (x >= threshold && y >= threshold) {
        local_A.resize(floor(N/q), vector<double> (floor(N/q)));
        for (int i=0; i<local_A.size(); i++) {
            for (int j=0; j<local_A[0].size(); j++)
                local_A[i][j] = proc_data[i*floor(N/q)+j];
        }
    }
    if (DEBUG) {
        cout << "----------------------------------------" << endl;
        printf("crank %i, local_A \n", crank);
        for (int i=0; i<local_A.size(); i++) {
            for (int j=0; j<local_A[0].size(); j++) {
                cout << local_A[i][j] << " ";
            } cout << endl;
        }
        cout << "----------------------------------------" << endl;
    }


    // Jacobi Algorithm starts
    starttime = MPI_Wtime();

    int count_itr = 0;
    bool convergence = false;

    // init x as 0 vector
    vector<double> stored_x;  // size based on #rows
    if (coords[0] == 0) {
        if (coords[1] < threshold)
            stored_x.resize(ceil(N/q), 0.0);
        else
            stored_x.resize(floor(N/q), 0.0);
    }

    while ((count_itr < MAX_ITR) && (convergence == false)) {

        vector<double> local_x;  // size based on #cols
        if (coords[0] < threshold)
            local_x.resize(ceil(N/q), 0.0);
        else
            local_x.resize(floor(N/q), 0.0);

        // Step 1: Rx
        // 1.1 MPI_Send() local_x from proc in col1 to proc in diagonal
        if (coords[0] == 0 || coords[0] == coords[1]) {
            for (int i = 1; i < q; i++) {
                if (coords[1] == i) {
                    if (coords[0] == 0) 
                        MPI_Send(stored_x.data(), stored_x.size(), MPI_DOUBLE, coords[1], 111, row_comm);
                    else
                        MPI_Recv(local_x.data(), local_x.size(), MPI_DOUBLE, 0, 111, row_comm, MPI_STATUS_IGNORE);
                }
            }
        }

        // 1.2 MPI_Bcast() local_b from proc in diagonal along its col comm
        if (crank == 0) local_x = stored_x;
        MPI_Bcast(local_x.data(), local_x.size(), MPI_DOUBLE, coords[0], col_comm);
        // if (DEBUG) {
        //     printf("crank %i, local_x: ", crank);
        //     for (int i = 0; i < local_x.size(); i++) cout << local_x[i] << " "; cout << endl;
        // }
        
        // 1.3 Local Matrix-Vector multiplication Rx
        vector<double> Rx;  // size based on #rows
        if (coords[1] < threshold)
            Rx.resize(ceil(N/q)), 0.0;
        else
            Rx.resize(floor(N/q), 0.0);

        for (int i = 0; i < local_A.size(); i++) {
            for (int j = 0; j < local_A[0].size(); j++) {
                if (coords[0] == coords[1]) {
                    if (i != j)
                        Rx[i] += local_A[i][j] * local_x[j];
                } else {
                    Rx[i] += local_A[i][j] * local_x[j];
                }
            }
        }
        // if (DEBUG) {
        //     printf("crank %i, Rx: ", crank);
        //     for (int i = 0; i < Rx.size(); i++) cout << Rx[i] << " "; cout << endl;
        // }

        // 1.4 MPI_Reduce() local Rx along row to proc in col1
        vector<double> reduced_Rx;  // size based on #rows
        if (coords[1] < threshold)
            reduced_Rx.resize(ceil(N/q), 0.0);
        else
            reduced_Rx.resize(floor(N/q), 0.0);

        MPI_Reduce(Rx.data(), reduced_Rx.data(), int(reduced_Rx.size()), MPI_DOUBLE, MPI_SUM, 0, row_comm);
        // if (DEBUG) {
        //     if (coords[0] == 0) {
        //         printf("crank %i, Reduced Rx: ", crank);
        //         for (int i = 0; i < reduced_Rx.size(); i++) cout << reduced_Rx[i] << " ";  cout << endl;
        //     }
        // }
        

        // Step 2: update x
        vector<double> new_stored_x;
        if (coords[0] == 0) {
            new_stored_x.resize(stored_x.size(), 0.0);
            for (int i = 0; i < stored_b.size(); i++)
                new_stored_x[i] = (1.0/stored_d[i]) * (stored_b[i] - reduced_Rx[i]);
            // if (DEBUG) {
            //     printf("crank %i, new_stored_x: ", crank);
            //     for (int i = 0; i < new_stored_x.size(); i++) cout << new_stored_x[i] << " "; cout << endl;
            // }
        }
        
        
        // Step 3: ||Ax||
        // 3.1 Global Ax
            // Send x to diagonal
        vector<double> new_local_x(local_x.size(), 0.0);
        if (coords[0] == 0 || coords[0] == coords[1]) {
            for (int i = 1; i < q; i++) {
                if (coords[1] == i) {
                    if (coords[0] == 0) 
                        MPI_Send(new_stored_x.data(), new_stored_x.size(), MPI_DOUBLE, coords[1], 111, row_comm);
                    else
                        MPI_Recv(new_local_x.data(), new_local_x.size(), MPI_DOUBLE, 0, 111, row_comm, MPI_STATUS_IGNORE);
                }
            }
        }
            // Broadcast x in col
        if (crank == 0) new_local_x = new_stored_x;
        MPI_Bcast(new_local_x.data(), new_local_x.size(), MPI_DOUBLE, coords[0], col_comm);
        // if (DEBUG) {
        //     printf("crank %i, new_local_x: ", crank);
        //     for (int i = 0; i < new_local_x.size(); i++)
        //         cout << new_local_x[i] << " "; cout << endl;
        // }
        
            // local Ax
        vector<double> Ax(Rx.size(), 0.0);
        for (int i = 0; i < local_A.size(); i++) {
            for (int j = 0; j < local_A[0].size(); j++)
                Ax[i] += local_A[i][j] * new_local_x[j];
        }
        // if (DEBUG) {
        //     printf("crank %i, Ax: ", crank);
        //     for (int i = 0; i < Ax.size(); i++) cout << Ax[i] << " "; cout << endl;
        // }

            // Reduce Ax to col1
        vector<double> reduced_Ax(reduced_Rx.size(), 0.0);
        MPI_Reduce(Ax.data(), reduced_Ax.data(), int(reduced_Ax.size()), MPI_DOUBLE, MPI_SUM, 0, row_comm);
        // if (DEBUG) {
        //     if (coords[0] == 0) {
        //         printf("crank %i, reduced_Ax: ", crank); 
        //         for (int i = 0; i < reduced_Ax.size(); i++) cout << reduced_Ax[i] << " "; cout << endl;
        //     }
        // }

        // 3.2 Local (Ax-b).(Ax-b) and MPI_AllReduce() within col1. Take sqrt() to get global ||Ax||
        if (coords[0] == 0) {
            double local_norm = 0, global_norm;
            for (int i = 0; i < reduced_Ax.size(); i++)
                local_norm += (reduced_Ax[i]-stored_b[i]) * (reduced_Ax[i]-stored_b[i]);

            MPI_Allreduce(&local_norm, &global_norm, 1, MPI_DOUBLE, MPI_SUM, col_comm);
            global_norm = sqrt(global_norm);
            // if (DEBUG) printf("crank %i, Ax_norm: %f \n", crank, global_norm);
        
            // 3.4 Compute convergence boolean and broadcast along row comm
            if (global_norm <= pow(10, -9))
                convergence = true;
        }
        MPI_Bcast(&convergence, 1, MPI_INT, 0, row_comm);

        // if (DEBUG) printf("crank %i, convergence bool: %i \n", crank, convergence);

        // set old x = new x for next iteration
        if (coords[0] == 0)
            stored_x = new_stored_x;

        count_itr++;
    }

    if (DEBUG) {
        if (crank == 0) cout << "num of itr: " << count_itr << endl;
        printf("crank %i, convergence: %i \n", crank, convergence);
        if (coords[0] == 0) {
            printf("crank %i, final x: ", crank);
            for (int i = 0; i < stored_x.size(); i++) cout << stored_x[i] << " "; cout << endl;
        }
    }


    // Gather all local x into crank 0
    vector<double> final_answer;
    if (coords[0] == 0) {
        final_answer.resize(N, 0.0);
        int prefix_recv = 0;
        vector<int> recv_counts(q), recv_displs(q);
        for (int i = 0; i < q; i++) {
            recv_counts[i] =  (i < threshold) ? ceil(N/q) : floor(N/q);
            recv_displs[i] = prefix_recv;
            prefix_recv += recv_counts[i];
        }

        MPI_Gatherv(stored_x.data(), stored_x.size(), MPI_DOUBLE, final_answer.data(), recv_counts.data(), recv_displs.data(), MPI_DOUBLE, 0, col_comm);

        if (DEBUG) {
            if (crank == 0) {
                printf("crank %i, reduce of final x\n", crank);
                cout << "recv_counts: ";
                for (int i=0; i<q; i++) cout << recv_counts[i] << " ";
                cout << endl << "recv_displs: ";
                for (int i=0; i<q; i++) cout << recv_displs[i] << " ";
                cout << endl;

                printf("crank %i, Final x: \n", crank);
                for (int i = 0; i < final_answer.size(); i++) cout << final_answer[i] << " "; cout << endl;
            }
        }
    }

    // STOP ALGO TIME
    endtime = MPI_Wtime();

    // write to output file
    if(crank == 0) {
        ofstream outfile;
        outfile.open(out_name, ofstream::out | ofstream::trunc);
        for (int i = 0; i < final_answer.size(); i++)
            outfile << setprecision(16) << fixed << final_answer[i] << " ";
        outfile.close();

        cout << "Runtime: " << endtime - starttime << endl;  // runtime
    }

    return MPI_Finalize();
}
