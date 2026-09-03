
#include <Rcpp.h>
#include <random>

using namespace std;
using namespace Rcpp;

// maximin_PTLHD (with convergence)
// Jaime Meyer Beilis Michel
// 7.1.2025 at Bloomington, Indiana


// Initialize the numeric matrix object.
NumericMatrix initialize_numMat(const vector<double>& sched, int n, int k, int index) {
    NumericMatrix result(n, k);

    const double* source_block = sched.data() + size_t(index) * n * k;
    memcpy( REAL(result),source_block, sizeof(double) * n * k );

    return result;
}


// Calculate the squared euclidean distance between two points
// of a given design 
// O(k) time

double dist_square(const double* D, int row1, int row2, int n, int k){
    double summation = 0.0; double diff; 
    for(int i = 0 ; i < k ; i++){
        diff = D[n * i + row1] - D[n * i+ row2];
        summation += diff * diff;
    }
    return summation;
}

inline double dist_square_block(const vector<double>& sched, int block, int row1, int row2, int n, int k){
    double summation = 0.0; double diff; 
    for(int i = 0 ; i < k ; i++){
        diff = sched[block + n * i + row1] - sched[block + n * i + row2];
        summation += diff * diff;
    }
    return summation;
}

// Calculate the measure (Energy) of a design D,
// by the average interpoint distance reciprocal
// O(n^2 k) time.

double phi(const double* D, int n, int k, double p){
    double summation = 0.0;
    for(int i = 0 ; i < n - 1 ; i++ ){
        for(int j = i + 1 ; j < n ; j++){
            // take the square root and power of -p at the same time
            summation += pow(dist_square(D, i, j,n, k), -p * 0.5);
        }
    }
    // average and the power of a large enough p
    return pow((2.0 / (n * (n-1))) * summation, 1.0/p);
}

// NOTES

// 7.9.2025
// Fixed major bug -> Negligence of global minima phi0. When accepted by probability, the global minima
//                    remained as the old phi value. Leading to an inexistent value
//                    Solved: Measure result was not accurate to phi function
//                    Solution:
//                    > Separation of acceptance logic
//                    > final check if measure is not global minima


// 7.9.2025 
// Fixed major bug -> Distances were being measured on row major instead of column major.
//                    Causing innacurate measures.
//                    Solved: Measure result was not accurate to phi function



// [[Rcpp::export]]
List pt_maximin_lhd(int n, int k, int M, int Nmax, int Nswap, double p, int tolerance){
    mt19937_64 RNG_engine{random_device{}()};
    uniform_int_distribution<int> distSwap(0, M-2);// for choosing random swap
    uniform_int_distribution<int> distCol(0, k-1); // for choosing random col
    uniform_int_distribution<int> distRow(0, n-1); // for choosing random row
    uniform_real_distribution<> dist(0.0, 1.0);    // for choosing uniformly distributed prob

    ////////////////////// Initialize random design ///////////////////////

    vector<int> klist(n);
    // fill 1:k
    iota(klist.begin(),klist.end(),1);

    // column major matrix
    double D0[n * k]; 

    for(int j = 0; j < k; j++){
        // shuffle klist
        for(int i = n - 1; i >= 0; --i){
            // shuffle the integers
            uniform_int_distribution<int> shuffler(0, i);
            swap(klist[i], klist[shuffler(RNG_engine)]);
            D0[j * n + i] = klist[i];
        }
    }
    
    
    /////////////////////// End random design initialization //////////////

    ////////////////////// Initialize temperatures  /////////////////////////

    double d2 = k/6.0 * n * (n+1);
    double delta0 = 1.0/sqrt(d2 - k) - 1/sqrt(d2);
    double T0 = -delta0 * (1.0/log(0.99));
    
    ///////////////////// Calculate initial phi //////////////////////////

    double phi0 = phi(D0, n, k, p);

    //////////////////// Make full replica schedule //////////////////////
    
    vector<double> design_sched;
    // allocate M nk doubles
    design_sched.resize(size_t(M) * n * k);

    vector<double> temp_sched;
    vector<double> phi_sched;


    for(int i = 0; i < M; i++){ // generate the schedules

        // Copy in the ith with the data from D0
        double * predestination = design_sched.data() + size_t(i) * n * k;
        // make a memory copy, this is the fastest possible performance for any matrix
        // cloning operation. It fits the usecase of lesser dimensions, for
        // which using numericMatrix is a lot of overkill.
        memcpy(predestination, D0, sizeof(double) * n * k);

        temp_sched.push_back(pow(T0,(M-i+1.0)));
        phi_sched.push_back(phi0);
    }

    //////////////////// End of replica schedule generation ////////////

    // convergence testing
    int unsuccessful_global_min = 0;
    int global_min_index = 0;
    int flag_false_best = false;

    // since here, phi0 will be utilized for global minimum. 
    // To ensure (somewhat) good use of space.

    ///////////////////// Run the PT algorithm ////////////////////
    for(int iter = 0; iter < Nmax; iter++){

        ///////////// Metropolis algorithm ///////////////////////
        for(int ii = 0; ii < M; ii++){
            
            // double* D = design_sched[ii * n * k + i * k + j];

            // choose perturbation rows and column
            int h = distCol(RNG_engine);
            int w = distRow(RNG_engine);
            int v;
            do { // ensure distinction between indices
                v = distRow(RNG_engine);
            } while (v == w);

            // recalculate phi
            double phi_original = phi_sched[ii];



            //////////////////// PHI RECALCULATION ///////////////////////////////
            
            double unwrap_phi = n * (n - 1) * 0.5 * pow(phi_original,p);
            double summation = 0.0;
            double diff_sqr1;
            double diff_sqr2;

            // keep track of the memory block of the matrix
            int block = (ii * n * k);
            
            int col_prefix = block + h * n;
            double minus_p_halves = (double)(-p * 0.5);
            for(int i = 0; i < n; i++){
                if( !(i == w || i == v) ){
                        double curr_element = design_sched[col_prefix + i];
                        diff_sqr1 = dist_square_block(design_sched,block, w, i, n, k);
                        diff_sqr2 = dist_square_block(design_sched,block, v, i, n, k);

                        double diff1 = design_sched[col_prefix + v] - curr_element;
                        double diff2 = design_sched[col_prefix + w] - curr_element;

                        double s_d = diff1 * diff1 - diff2 * diff2;
                        summation += (
                            pow(diff_sqr1 + s_d, minus_p_halves) - pow(diff_sqr1, minus_p_halves)
                          + pow(diff_sqr2 - s_d,minus_p_halves)  - pow(diff_sqr2,  minus_p_halves)
                        );
                    }
            }

            double phi_try = pow((unwrap_phi + summation) * 2.0 / ( n * (n - 1) ), 1.0/p);


            

            /////////////////// PHI RECALCULATION END ////////////////////////////////


            // if the recalculated phi try is better than the original
            // or alternatively the probability of acceptance is high enough 
            // compared to a uniformly distributed random probability
            if(
                phi_try < phi_original
            )   
            {
                // accept the perturbation of the design
                swap(design_sched[block +  h * n + w], design_sched[block + h * n + v]);
                // the phi designated becomes that of the perturbed design
                phi_sched[ii] = phi_try;


                // check if phi_try is the global minimum
                if(phi_try < phi0){
                    // if true, update global minimum
                    phi0 = phi_try;
                    unsuccessful_global_min = 0;
                    global_min_index = ii;
                    flag_false_best = false;
                }

            }else if( dist(RNG_engine) <= exp(-(phi_try - phi_original)/temp_sched[ii])){
                swap(design_sched[block +  h * n + w], design_sched[block + h * n + v]);
                phi_sched[ii] = phi_try;

                if(global_min_index == ii){
                    phi0 = phi_try;
                    flag_false_best = true;
                }
            }

        }

        ////////////// Swap configuration ///////////////////////
        if((iter + 1) % Nswap == 0){

            int q = distSwap(RNG_engine);
            double phi1 = phi_sched[q];
            double phi2 = phi_sched[q+1];
            double t1 = temp_sched[q];
            double t2 = temp_sched[q+1];

            // Since we are keeping three lists instead of 1, Decide on swapping
            // the temperatures. If I am not wrong, it should be mathematically equivalent
            // to swapping the designs. Regardless of having broken the original order
            // of T1 < T2 < T3 < ... < TM, just an irrelevant -1 swap to worry about.
            if(dist(RNG_engine) <= exp((phi2 - phi1) * (1/t2 - 1/t1))){
                swap(temp_sched[q], temp_sched[q+1]);
            }

        }
        /////////////// End of configuration swap ///////////////


        // unsuccesful_global_minimum check
        if(++unsuccessful_global_min > tolerance){
            Nmax = iter;
            break;
        }
    }

    // Rule engforced. If not true global minimum, search list again.
    if(flag_false_best){
        
        for(int i = 0 ; i < M ; i++){
            double phi_curr = phi_sched[i];
            if(phi_curr < phi0) {
                phi0 = phi_curr;
                global_min_index = i;
            }
        }
    }

    //////////////////// End of the PT algorithm //////////////////
    return List::create(
            Named("Design") = initialize_numMat(design_sched,n,k,global_min_index),
            Named("Measure") = phi0,
            Named("t0") = temp_sched,
            Named("ntotal") = Nmax
    );
}
// End of code, why are you still here?
