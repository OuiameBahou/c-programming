#include "EX3.h"

person createpers(){
    person P;
    printf("person's name: ");
    scanf("%s",P.name);
    printf("person's phone number: ");
    scanf("%s",P.phoneNo);
    return P;
}

void addpers(person P, list* L){
    if(L->Num == 8){
        printf("Contact book full, do you want to remove the first entry ? (yes or no) ");
        char ans[4];
        scanf("%s",ans);
        if(strcmp(ans, "yes") == 0){
            cell* temp = L->head;
            L->head = L->head->following;
            L->head->previous = NULL;
            free(temp);
            cell* add = (cell*)malloc(sizeof(cell));
            add->X = P;
            add->following = NULL;
            add->previous = L->tail;
            L->tail->following = add;
            L->tail = add;
            printf("Person successfully added to the contanct book with id %u\n", L->Num);
        }
        else printf("Opperation interupted.\n");
    }
    else{
        if(L->Num == 0){
            cell* add = (cell*)malloc(sizeof(cell));
            add->X = P;
            add->following = NULL;
            add->previous = NULL;
            L->head = add;
            L->tail = add;
            L->Num++;
        }
        else{
            cell* add = (cell*)malloc(sizeof(cell));
            add->X = P;
            add->following = NULL;
            add->previous = L->tail;
            L->tail->following = add;
            L->tail = add;
            L->Num++;
        }
        printf("Person successfully added to the contanct book with id %u\n", L->Num);

    }
}

void showlist(list L){
    if(L.Num == 0) printf("Empty conctact book.\n");
    else{
        int i = 1;
        while(L.head != NULL){
            printf("Name: %s\nPhone number: %s\nId: %u\n\n", L.head->X.name, L.head->X.phoneNo, i);
            i++;
            L.head = L.head->following;
        }
    }
}

void destroy(list* L){
    while(L->Num != 0){
        cell* temp = L->head;
        L->head = L->head->following;
        free(L->head);
        L->Num--;
    }
}