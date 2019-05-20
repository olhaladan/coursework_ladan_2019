#include <iostream>
#include <fstream>
#include <math.h>
#include <vector>
#include <chrono>
#include <omp.h>
#include <string>
using namespace std;

class Calculate
{
    private:
        static const long N_X = 3; 	//2	 3  10  100
        static const long N_T = 19;	//9	 19	 250  50000

        static constexpr double H = 1.0 / (N_X - 1);
        static constexpr double TAU = 1.0 / (N_T - 1);
        static constexpr double SIGMA = TAU / (H * H);

        static constexpr double C = 2; //constexp-они инициализируются до создания любых потоков. В результате их доступ никогда не требует синхронизации
        static constexpr double LAMBDA = -0.5;
        static constexpr double A = 1.5;
        static constexpr double B = 4;

        //точний розв'язок
        double getExactSolution(double x, double t)
        {
            return pow(C * exp(-3 * LAMBDA / A * (x + LAMBDA * t)) - B / (4 * LAMBDA), -1.0/3.0); 
        }

        //початкові умови
        //t=0
        void getStartOmega(vector<double> & vec)
        {
            vec.resize(N_X);
            #pragma omp parallel for
                for (long i=0;i<N_X;i++)
                    vec[i] = getExactSolution(VEC_X[i],0);
        }

        //граничні умови
        //x=0
        double getLeftBorderValue(double t)
        {
            return pow(C * exp(-3 * LAMBDA / A * LAMBDA * t) - B / (4 * LAMBDA), -1.0/3.0);
        }
        //x=1
        double getRightBorderValue(double t)
        {
            return pow(C * exp(-3 * LAMBDA / A * (1 + LAMBDA * t) ) - B / (4 * LAMBDA), -1.0/3.0);
        }
		//рахуємо не граничні умови 
        void getNonBorderValues(vector<double> prevOmega, vector<double> & vec)
        {
            #pragma omp parallel for
            	for(long i=1;i<N_X-1;i++)
                	vec[i] = prevOmega[i] + TAU * ( A * ( (prevOmega[i-1] - 2*prevOmega[i] + prevOmega[i+1]) / (H*H) ) + B*pow(prevOmega[i],3)*(prevOmega[i+1] - prevOmega[i-1])/(2*H));
        }
        	
        

        public:
            vector<double> VEC_X;
            vector<double> VEC_T;

            Calculate()
            {
                if ( TAU / (H*H) >= 0.5 ) //?
                {
                    cerr << "Unsolve with those steps.\n";
                    exit(1);
                }
                //сетка(шаг)
                for (long i=0;i<N_T;i++)
                    VEC_T.push_back(i * TAU);
                for (long i=0;i<N_X;i++)
                    VEC_X.push_back(i * H);
            }

            //рахуємо граничні умови
            void getBorder(vector<vector<double>> & omega)
            {
                omega.resize(N_T);
                getStartOmega(omega[0]);
                #pragma omp parallel for
                    for (long k=1;k<N_T;k++)
                    {
                        omega[k].resize(N_X);
                        omega[k][0] = getLeftBorderValue(VEC_T[k]);
                        omega[k][N_X-1] = getRightBorderValue(VEC_T[k]);
                    }
            }
            
            //рахуємо омега
            void getOmega(vector<vector<double>> & omega)
            {
                for (long k=1;k<N_T;k++)
                    getNonBorderValues(omega[k-1], omega[k]);
            }
            
            
    		void getExactValue(vector<vector<double>> & apr)
			{
				apr.resize(N_T);
					for (long k=0;k<N_T;k++)
	            	{   
		            	apr[k].resize(N_X);
		               	for (long i=0;i<N_X;i++)
		               	{
		               		apr[k][i] = getExactSolution(VEC_X[i], VEC_T[k]);
		                }
	            	}
			}

            //рахуємо похибку
            void getErrors(vector<vector<double>> & calculated, double & absoluteMax, double & relativeMax)
            {
                for (long k=1;k<N_T;k++)
                    {
                        #pragma omp parallel for shared(absoluteMax, relativeMax)
                            for (long i=1;i<N_X-1;i++)
                            {
                                double real = getExactSolution(VEC_X[i], VEC_T[k]);
                                double approxim = fabs(calculated[k][i] - real);
                                if (approxim > absoluteMax) absoluteMax = approxim;
                                if (approxim/real * 100 > relativeMax) relativeMax = approxim/real * 100;
                            }
                        
                    }
            }
};

void print(vector<double> & vec, string filename)
{
    ofstream f;
    f.open(filename);
    f << "{";
    for (int i=0;i<vec.size()-1;i++)
        f << vec[i] << ", ";
    f << vec[vec.size()-1] << "};";
    f.close();
}

void print(vector<vector<double>> & vec, string filename, vector<double> X,vector<double> T)
{
    ofstream f;
    f.open(filename);
    
    f << "                      ";
    for (int w=0;w<X.size();w++)
    f << X[w] << "        ";
	f << endl;

    for (int i=0;i<vec.size();i++)
    {
		f<< T[i] << "           " ;
    	f << "{";
        for (int j=0;j<vec[i].size()-1;j++)
            f << vec[i][j] << ", ";
        
        f << vec[i][vec[i].size()-1] << "}," << endl;
    }
    f.close();
}

int main()
{
    //створюємо екземпляр класу та необхідні змінні для розрахунків
    Calculate calc;
    vector<vector<double>> omega;
    vector<vector<double>> ev;
    double absoluteMax = 0, relativeMax = 0;
    int max_th = omp_get_max_threads();

    //-----послідовні розрахунки-----
    omp_set_num_threads(1);
    //час початку
    auto start_t = chrono::system_clock::now();
    //обчислимо початкові умови та омегу
    calc.getBorder(omega);
    calc.getOmega(omega);
    //обчислимо похибки
    calc.getErrors(omega, absoluteMax, relativeMax);
    //час закінчення
    auto end_t = chrono::system_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds> (end_t - start_t).count();
    cout << "Sequential calculations:" << endl << "  Runtime: " << duration << endl << "  Absolute max: " << absoluteMax << endl << "  Relative max: " << relativeMax << endl;

    //підготуємо змінні для подальшої роботи
    omega.clear();
    absoluteMax = 0;
    relativeMax = 0;

    //-----паралельні розрахунки-----
    omp_set_num_threads(max_th);
    //час початку
    start_t = chrono::system_clock::now();
    calc.getBorder(omega);
    calc.getOmega(omega);
    //обчислимо похибки
    calc.getErrors(omega, absoluteMax, relativeMax);
    //час закінчення
    end_t = chrono::system_clock::now();
    duration = chrono::duration_cast<chrono::milliseconds> (end_t - start_t).count();
    cout << "Parallel calculations:" << endl << "  Runtime: " << duration << endl << "  Absolute max: " << absoluteMax << endl << "  Relative max: " << relativeMax << endl;
	cout << "max_th:" << max_th << endl;
    //друкуємо результати
    
    print(omega, "omega.txt", calc.VEC_X, calc.VEC_T);
    print(calc.VEC_T, "t.txt");
    print(calc.VEC_X, "x.txt");
    
    calc.getExactValue(ev);
    print(ev, "ev.txt", calc.VEC_X, calc.VEC_T);
    
    return 0;
}
