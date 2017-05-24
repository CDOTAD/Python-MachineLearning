# -*- coding:utf-8 -*-

import math
import pickle


class NaiveItemCollaborativeFiltering:

    def __init__(self,user,data,k = 10,similar_matrix=None):

        self.user_recommend = user

        self.data_set = data

        self.topK = k

        if type(similar_matrix) == type("str"):

            self.similar_matrix = pickle.load(open(similar_matrix,'rb'))

        else:

            self.similar_matrix = None

        return

    def item_similarity(self,store=False):

        train = self.data_set

        scale_matrix = dict()

        neighbor_matrix = dict()

        for u,items in train.items():

            for i in items:

                if i not in neighbor_matrix:

                    neighbor_matrix[i] = 0

                neighbor_matrix[i] += 1

                if i not in scale_matrix:

                    scale_matrix[i] = dict()
                for j in items:

                    if i == j:

                        continue

                    if j not in scale_matrix[i]:

                        scale_matrix[i][j] = 0

                    scale_matrix[i][j] += 1

        similar_matrix=dict()

        for i,related_items in scale_matrix.items():

            if i not in similar_matrix:

                similar_matrix[i] = dict()

            for j,cij in related_items.items():

                similar_matrix[i][j] = cij / math.sqrt(neighbor_matrix[i] * neighbor_matrix[j])

        self.similar_matrix=similar_matrix

        if store:

            store_file=open('item_similar_matrix','wb')

            pickle.dump(similar_matrix,store_file)

        return similar_matrix

    def recommend(self):

        if self.similar_matrix is None:

            self.similar_matrix = self.item_similarity()

        rank = dict()

        ru = self.data_set[self.user_recommend]

        for i in ru:

            w_item=self.similar_matrix[self.user_recommend]

            for j,wj in sorted(w_item.items(),key=lambda w_item:w_item[1],reverse=True)[0:self.topK]:

                if j in ru:

                    continue

                if j not in rank:

                    rank[j] = 0

                rank[j] += wj

        recommend_list = sorted(rank.items(),key=lambda rank:rank[1],reverse=True)[0:self.topK]

        return recommend_list



