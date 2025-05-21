#include "socket_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <errno.h>

int setup_server(int port) {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in addr;

    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = INADDR_ANY;
    bind(server_fd, (struct sockaddr*)&addr, sizeof(addr));
    listen(server_fd, 1);

    return server_fd;
}

int accept_client(int server_fd) {
    struct sockaddr_in client_addr;
    socklen_t client_len = sizeof(client_addr);
    return accept(server_fd, (struct sockaddr*)&client_addr, &client_len);
}

int connect_to_server(const char* ip, int port) {
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in addr;

    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    inet_pton(AF_INET, ip, &addr.sin_addr);
    
    if (connect(sockfd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("connect");
        exit(1);
    }

    return sockfd;
}

int send_all(int sockfd, const void* data, int size) {
    const char* ptr = (const char*)data;
    int total = 0;
    while (total < size) {
        int sent = send(sockfd, ptr + total, size - total, 0);
        if (sent <= 0) return -1;
        total += sent;
    }
    return total;
}

int recv_all(int sockfd, void* buffer, int size) {
    char* ptr = (char*)buffer;
    int total = 0;
    while (total < size) {
        int recvd = recv(sockfd, ptr + total, size - total, 0);
        if (recvd <= 0) return -1;
        total += recvd;
    }
    return total;
}

void socket_close(int sockfd) {
    close(sockfd);
}