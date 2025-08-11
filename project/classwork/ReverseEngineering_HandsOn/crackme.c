
#include <stdio.h>
#include <string.h>

int main() {
    char input[20];
    printf("Enter the password: ");
    scanf("%19s", input);
    if (strcmp(input, "cyber123") == 0) {
        printf("Access Granted!\n");
    } else {
        printf("Access Denied.\n");
    }
    return 0;
}