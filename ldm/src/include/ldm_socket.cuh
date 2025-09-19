#ifndef LDM_SOCKET_CUH
#define LDM_SOCKET_CUH

#include <iostream>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>
#include <vector>
#include <sstream>
#include <iomanip>
#include <limits>
#include <algorithm>
#include "ldm_config.cuh"

// External config reference
extern ConfigReader g_config;

//#define PORT 8080 // Now loaded from config file

void send_gamma_dose_matrix(float* h_gamma_dose, int rows, int cols) {
    int server_fd, new_socket;
    struct sockaddr_in address;
    int opt = 1;
    int addrlen = sizeof(address);

    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        perror("Socket failed");
        exit(EXIT_FAILURE);
    }

    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt))) {
        perror("setsockopt");
        exit(EXIT_FAILURE);
    }
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(g_config.getInt("socket_port", 8080));
    //printf("a\n");

    if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) < 0) {
        perror("Bind failed");
        exit(EXIT_FAILURE);
    }
    if (listen(server_fd, 3) < 0) {
        perror("Listen failed");
        exit(EXIT_FAILURE);
    }

    std::cout << "Waiting for connection..." << std::endl;
    if ((new_socket = accept(server_fd, (struct sockaddr*)&address, (socklen_t*)&addrlen)) < 0) {
        perror("Accept failed");
        exit(EXIT_FAILURE);
    }
    //printf("b\n");


    int matrix_size = rows * cols;
    send(new_socket, h_gamma_dose, matrix_size * sizeof(float), 0);
    std::cout << "Gamma dose matrix sent successfully!" << std::endl;

    close(new_socket);
    close(server_fd);
}

void receive_tmp_states(float* tmp_states, int size) {
    int server_fd, new_socket;
    struct sockaddr_in address;
    int opt = 1;
    int addrlen = sizeof(address);

    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        perror("Socket failed");
        exit(EXIT_FAILURE);
    }

    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt))) {
        perror("setsockopt");
        exit(EXIT_FAILURE);
    }
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(g_config.getInt("socket_port", 8080));

    if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) < 0) {
        perror("Bind failed");
        exit(EXIT_FAILURE);
    }
    if (listen(server_fd, 3) < 0) {
        perror("Listen failed");
        exit(EXIT_FAILURE);
    }

    std::cout << "Waiting for connection..." << std::endl;
    if ((new_socket = accept(server_fd, (struct sockaddr*)&address, (socklen_t*)&addrlen)) < 0) {
        perror("Accept failed");
        exit(EXIT_FAILURE);
    }

    int data_size;
    recv(new_socket, &data_size, sizeof(int), 0);

    std::vector<float> buffer(data_size / sizeof(float));
    recv(new_socket, buffer.data(), data_size, 0);

    std::memcpy(tmp_states, buffer.data(), data_size);

    std::cout << "Received tmp_states data successfully!" << std::endl;

    close(new_socket);
    close(server_fd);
}

void send_concentration_matrix_to_eki(float* h_concentration_matrix, int ensemble_size, int num_receptors, int time_intervals) {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        std::cerr << "Socket creation failed for concentration matrix transmission.\n";
        return;
    }

    sockaddr_in server_addr;
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(g_config.getInt("socket_port", 8080));  // Use same port as EKI
    std::string server_ip = g_config.getString("server_ip", "127.0.0.1");
    inet_pton(AF_INET, server_ip.c_str(), &server_addr.sin_addr);

    std::cout << "Connecting to EKI for concentration matrix transmission..." << std::endl;
    std::cout << "Matrix dimensions: " << ensemble_size << " ensembles × " << num_receptors << " receptors × " << time_intervals << " time intervals" << std::endl;

    if (connect(sock, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        std::cerr << "Connection to EKI failed.\n";
        close(sock);
        return;
    }

    // Send dimensions first (3 integers: ensemble_size, num_receptors, time_intervals)
    int dimensions[3] = {ensemble_size, num_receptors, time_intervals};
    if (send(sock, dimensions, 3 * sizeof(int), 0) < 0) {
        std::cerr << "Failed to send dimensions.\n";
        close(sock);
        return;
    }

    // Calculate total data size
    int total_elements = ensemble_size * num_receptors * time_intervals;
    int data_size = total_elements * sizeof(float);
    
    std::cout << "Sending " << total_elements << " concentration values (" << data_size << " bytes)" << std::endl;

    // Send the concentration matrix data
    if (send(sock, h_concentration_matrix, data_size, 0) < 0) {
        std::cerr << "Failed to send concentration matrix data.\n";
        close(sock);
        return;
    }

    std::cout << "Concentration matrix sent to EKI successfully!" << std::endl;
    std::cout << "Data format: [ensemble][receptor][time] with " << total_elements << " total values" << std::endl;

    // Optional: Log sample data for debugging
    std::cout << "Sample concentration values (first 5): ";
    for (int i = 0; i < std::min(5, total_elements); i++) {
        std::cout << h_concentration_matrix[i] << " ";
    }
    std::cout << std::endl;

    close(sock);
}

// Legacy function kept for compatibility - now requires dimensions as parameters
void send_gamma_dose_matrix_ens(float* h_gamma_dose, int ensemble_size, int num_receptors, int time_intervals) {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        std::cerr << "Socket creation failed.\n";
        return;
    }

    sockaddr_in server_addr;
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(g_config.getInt("tcp_port", 12345));
    std::string server_ip = g_config.getString("server_ip", "127.0.0.1");
    inet_pton(AF_INET, server_ip.c_str(), &server_addr.sin_addr);

    if (connect(sock, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        std::cerr << "Connection failed.\n";
        close(sock);
        return;
    }

    // Send data as CSV string with newlines
    std::ostringstream data_stream;
    data_stream << std::setprecision(std::numeric_limits<float>::max_digits10);
    std::cout << std::setprecision(std::numeric_limits<float>::max_digits10);

    for (int ens = 0; ens < ensemble_size; ++ens) {
        for (int t = 0; t < time_intervals; ++t) {
            data_stream << ens << "," << t + 1;
            for (int r = 0; r < num_receptors; ++r) {
                int idx = ens * num_receptors * time_intervals + t * num_receptors + r;
                data_stream << "," << h_gamma_dose[idx];
            }
            data_stream << "\n";
        }
    }

    std::string data_str = data_stream.str();
    send(sock, data_str.c_str(), data_str.size(), 0);
    close(sock);
}

#endif // LDM_SOCKET_CUH
