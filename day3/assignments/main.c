#include "EX3.c"

int main(){
    list P = {NULL, NULL, 0};
    printf("Welcome to your contact book, type \"ADD \" to add a person, \"LIST\" to show all your contacts, \"QUIT\" to escape the programe.\n");
    while(1){
        char cmd[4];
        scanf("%s",cmd);
        if(strcmp(cmd, "ADD") == 0){
            person added = createpers();
            addpers(added, &P);
        }
        if(strcmp(cmd, "LIST") == 0) showlist(P);
        if(strcmp(cmd, "QUIT") == 0){
            printf("Goodbye !");
            destroy(&P);
            return 0;
        }
    }
}