  SEQUENTIAL PROGRAM WITH FUNCTIONS
#include <iostream>
#include <cstdlib>
#include <ctime>
void printHeader() {
    std::cout << "\t\tSequential Program with Functions" << std::endl;
    std::cout << "\t\t--------------------------------" << std::endl << std::endl;
}
void generateRandomArray(int* arr, int n) {
    for (int i = 0; i < n; ++i) {
        arr[i] = rand() % 100; // Random integer between 0 and 99
    }
}
void printArray(int* arr, int n) {
    std::cout << "\tArray elements: ";
    for (int i = 0; i < n; ++i) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl << std::endl;
}
int findSum(int* arr, int n) {
    int sum = 0;
    for (int i = 0; i < n; ++i) {
        sum += arr[i];
    }
    return sum;
}
bool searchKey(int* arr, int n, int key) {
    for (int i = 0; i < n; ++i) {
        if (arr[i] == key) {
            return true;
        }
    }
    return false;
}
int main() {
    printHeader();
    srand(static_cast<unsigned int>(time(0)));
    int n;
    std::cout << "\tEnter the size of the array: ";
    std::cin >> n;
    std::cout << std::endl;
    int* arr = new int[n];
    generateRandomArray(arr, n);
    std::cout << "\tGenerated Array:" << std::endl;
    printArray(arr, n);
    int sum = findSum(arr, n);
    std::cout << "\tSum of elements: " << sum << std::endl << std::endl;
    int key;
    std::cout << "\tEnter the key to search: ";
    std::cin >> key;
    std::cout << std::endl;
    bool found = searchKey(arr, n, key);
    if (found) {
        std::cout << "\tKey found in the array!" << std::endl;
    } else {
        std::cout << "\tKey not found in the array!" << std::endl;
    }
    delete[] arr;
    return 0;
}


THREAD BASED PROGRAM
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <thread>
#include <vector>
void printHeader() {
    std::cout << "\t\tMultithreaded Program with Functions" << std::endl;
    std::cout << "\t\t-----------------------------------" << std::endl << std::endl;
}
void generateRandomArray(int* arr, int n) {
    for (int i = 0; i < n; ++i) {
        arr[i] = rand() % 100; // Random integer between 0 and 99
    }
}
void printArray(int* arr, int n) {
    std::cout << "\tArray elements: ";
    for (int i = 0; i < n; ++i) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl << std::endl;
}
void findSum(int* arr, int start, int end, int* result) {
    *result = 0;
    for (int i = start; i < end; ++i) {
        *result += arr[i];
    }
}
void searchKey(int* arr, int start, int end, int key, bool* result) {
    *result = false;
    for (int i = start; i < end; ++i) {
        if (arr[i] == key) {
            *result = true;
            break;
        }
    }
}
int main() {
    printHeader();
    srand(static_cast<unsigned int>(time(0)));
    int n;
    std::cout << "\tEnter the size of the array: ";
    std::cin >> n;
    std::cout << std::endl;
    int* arr = new int[n];
    generateRandomArray(arr, n);
    std::cout << "\t\tGenerated Array:" << std::endl;
    printArray(arr, n);
    int numThreads = 4;
    std::vector<std::thread> threads;
    int partSize = n / numThreads;
    int* partialSums = new int[numThreads];

    std::cout << "\tCalculating sum of elements using " << numThreads ;
    std::cout<< " threads..." << std::endl;
    for (int i = 0; i < numThreads; ++i) {
        threads.push_back(std::thread(findSum, arr, i * partSize, (i + 1) 
* partSize, &partialSums[i]));
    }
    for (auto& t : threads) {
        t.join();
    }
    int totalSum = 0;
    for (int i = 0; i < numThreads; ++i) {
        totalSum += partialSums[i];
    }
    std::cout << "\tSum of elements: " << totalSum << std::endl << std::endl;
    threads.clear();
    bool* foundResults = new bool[numThreads];
    int key;
    std::cout << "\tEnter the key to search: ";
    std::cin >> key;
    std::cout << std::endl;
    std::cout << "\tSearching for key using " << numThreads;
    std::cout<< " threads..." << std::endl;
    for (int i = 0; i < numThreads; ++i) {
        threads.push_back(std::thread(searchKey, arr, i * partSize, (i + 1) 
* partSize, key, &foundResults[i]));
    }
    for (auto& t : threads) {
        t.join();
    }
    bool found = false;
    for (int i = 0; i < numThreads; ++i) {
        if (foundResults[i]) {
            found = true;
            break;
        }
    }
    if (found) {
        std::cout << "\tKey found in the array!" << std::endl;
    } else {
        std::cout << "\tKey not found in the array!" << std::endl;
    }
    delete[] arr;
    delete[] partialSums;
    delete[] foundResults;
    return 0;
}
