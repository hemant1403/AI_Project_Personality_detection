
import numpy as np  
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import copy
import math

def intro():

    #To Do:- Explain all personality types and give basic intro of program
    #To Do:- Explain O.C.E.A.N personality test
    # Cluster and unki averages ka score Code_AI_Project mein hai usko run kr ke end mein sbke avgs print ho rhe
    # Un averages se hr ko roughly koi personality type de do like one slighty more extroverted one slightly less open
    print("This Personality test is based on OCEAN personality test \n")
    print("The ocean personality test is based on the five-factor model, an empirical concept in psychology that measures five pre-dominant personality traits: openness, conscientiousness, extroversion, agreeableness, and neuroticism, making the acronym OCEAN. ")
    print("\n\nOpenness: (opn)(group 0) Openness is a characteristic that includes imagination an eagerness to learn and experience new things is particularly high for this personality trait.\n")
    print("Conscientiousness: (csn) (group 1) Conscientiousness is a trait that includes high levels of thoughtfulness, good impulse control, and goal-directed behaviours. \n")
    print("Extraversion: (ext)(group 2) extraversion is easily identifiable and widely recognisable as “someone who gets energised in the company of others.\n")
    print("Agreeableness: (agr) (group 3) People who exhibit high agreeableness will show signs of trust, altruism, kindness, and affection.\n")
    print("Neuroticism: (est) (group 4) Neuroticism is characterised by sadness, moodiness, and emotional instability. Often mistaken for anti-social behaviour")
    print("")
    print("Welcome to this personality detection \n")
    print("Answer all questions using numbers 1-5 \n")
    print("1 - Highly Disagree, 3 - Neutral, 5 -Highly Agree \n\n")

    answer_lst = []         #to store all answers

    questions_lst = [
        "I am the life of the party:- ", "I don't talk a lot:- ", "I feel comfortable around people:- ", "I keep in the background:- ",
        "I start conversations:- ", "I have little to say:- ", "I talk to a lot of different people at parties:- ",
        "I don't like to draw attention to myself:- ", "I don't mind being the center of attention:- ", "I am quiet around strangers:- ", 
        "I get stressed out easily:- ", "I am relaxed most of the time:- ", "I worry about things:- ", "I seldom feel blue:- ",
        "I am easily disturbed:- ", "I get upset easily:- ", "I change my mood a lot:- ", "I have frequent mood swings:- ", 
        "I get irritated easily:- ", "I often feel blue:- ", "I feel little concern for others:- ", "I am interested in people:- ",
        "I insult people:- ", "I sympathize with others' feelings:- ", "I am not interested in other people's problems:- ",
        "I have a soft heart:- ", "I am not really interested in others:- ", "I take time out for others:- ", 
        "I feel others' emotions:- ", "I make people feel at ease:- ", "I am always prepared:- ", "I leave my belongings around:- ",
        "I pay attention to details:- ", "I make a mess of things:- ", "I get chores done right away:- ", 
        "I often forget to put things back in their proper place:- ", "I like order:- ", "I shirk my duties:- ", "I follow a schedule:- ",
        "I am exacting in my work:- ", "I have a rich vocabulary:- ", "I have difficulty understanding abstract ideas:- ",
        "I have a vivid imagination:- ", "I am not interested in abstract ideas:- ", "I have excellent ideas:- ", 
        "I do not have a good imagination:- ", "I am quick to understand things:- ", "I use difficult words:- ", 
        "I spend time reflecting on things:- ","I am full of ideas:- "
]       
    #List of all questions

    for i in range(len(questions_lst)):
        x = int(input(questions_lst[i]))
        answer_lst.append(x)
        if x > 5 or x < 0:
            print("You have entered an invalid input \n Program will be restarted \n")
            intro()

    classification(answer_lst)

def classification(to_fit):

    df = pd.read_csv("data-final.csv", sep='\t')
    df.drop(df.columns[50:], axis=1, inplace=True)
    df_copy = copy.deepcopy(df)
    df = df.replace(0, np.nan).dropna(axis=0).reset_index(drop=True)    # remove all data that may contain a question 
                                                                        # whose answer is either False or not in accepted range

    ext = list(df.columns[0:10])
    est = list(df.columns[10:20])
    agr = list(df.columns[20:30])
    csn = list(df.columns[30:40])
    opn = list(df.columns[40:50])

    df['ext_rank'] = df[ext].sum(axis=1)/10     # summation of extrovert personality trait
    df['est_rank'] = df[est].sum(axis=1)/10     # summation of neurotic personality trait
    df['agr_rank'] = df[agr].sum(axis=1)/10     # summation of agreeableness personality trait
    df['csn_rank'] = df[csn].sum(axis=1)/10     # summation of conscientiousness personality trait
    df['opn_rank'] = df[opn].sum(axis=1)/10     # summation of openness personality trait

    kmeans_df_0 =  df.filter({'ext_rank', 'est_rank', 'agr_rank', 'csn_rank', 'opn_rank'})
    kmeans = KMeans(n_clusters=5)
    k_fit = kmeans.fit(df)
    predictions = k_fit.labels_
    kmeans_df_0['Clusters'] = predictions
    # classification on the basis of ranks into 5 types

    kmeans_df_1 = df
    kmeans = KMeans(n_clusters=5)
    k_fit = kmeans.fit(df)
    predictions = k_fit.labels_
    kmeans_df_1['Clusters'] = predictions
    # classify on basis of all questions plus ranks of each personality trait into 5 types

    c10_df = kmeans_df_1.loc[df['Clusters']==0]
    c10_df = c10_df.filter({'ext_rank', 'est_rank', 'agr_rank', 'csn_rank', 'opn_rank'})

    c11_df = kmeans_df_1.loc[df['Clusters']==1]
    c11_df = c11_df.filter({'ext_rank', 'est_rank', 'agr_rank', 'csn_rank', 'opn_rank'})

    c12_df = kmeans_df_1.loc[df['Clusters']==2]
    c12_df = c12_df.filter({'ext_rank', 'est_rank', 'agr_rank', 'csn_rank', 'opn_rank'})

    c13_df = kmeans_df_1.loc[df['Clusters']==3]
    c13_df = c13_df.filter({'ext_rank', 'est_rank', 'agr_rank', 'csn_rank', 'opn_rank'})

    c14_df = kmeans_df_1.loc[df['Clusters']==4]
    c14_df = c14_df.filter({'ext_rank', 'est_rank', 'agr_rank', 'csn_rank', 'opn_rank'})

    pca_1 = PCA(n_components=2).fit_transform(kmeans_df_1)
    pca_df_1 =  pd.DataFrame(data=pca_1, columns=['COMP3', 'COMP4'])
    pca_df_1['Clusters'] = predictions
    graph1 = sns.scatterplot(data=pca_df_1, x='COMP3', y='COMP4', hue='Clusters', palette='tab10')
    # print graph of df_1 classifiaction; that forms clusters 

    print(graph1)

    centers = np.array(kmeans.cluster_centers_)
    centers = centers.tolist()

    detection(to_fit, centers, c10_df, c11_df, c12_df, c13_df, c14_df)

def detection(to_fit, centers, c10_df, c11_df, c12_df, c13_df, c14_df):

    ext_score = sum(to_fit[0:10])
    ext_score /= 10

    est_score = sum(to_fit[10:20])
    est_score /= 10

    agr_score = sum(to_fit[20:30])
    agr_score /= 10

    csn_score = sum(to_fit[30:40])
    csn_score /= 10

    opn_score = sum(to_fit[40:50])
    opn_score /= 10

    to_fit.append(ext_score)
    to_fit.append(est_score)
    to_fit.append(agr_score)
    to_fit.append(csn_score)
    to_fit.append(opn_score)
    # append all answers + ranks into a new list

    min_cluster_0 = 0
    min_cluster_1 = 0
    min_cluster_2 = 0
    min_cluster_3 = 0
    min_cluster_4 = 0

    for i in range(len(centers[0])):
        min_cluster_0 += math.sqrt(abs(centers[0][i]**2 - to_fit[i]**2))

    for i in range(len(centers[1])):
        min_cluster_1 += math.sqrt(abs(centers[1][i]**2 - to_fit[i]**2))

    for i in range(len(centers[2])):
        min_cluster_2 += math.sqrt(abs(centers[2][i]**2 - to_fit[i]**2))

    for i in range(len(centers[3])):
        min_cluster_3 += math.sqrt(abs(centers[3][i]**2 - to_fit[i]**2))

    for i in range(len(centers[4])):
        min_cluster_4 += math.sqrt(abs(centers[4][i]**2 - to_fit[i]**2))

    # calculate distance of given data from each centroid of the cluster

    prediciton(min_cluster_0, min_cluster_1, min_cluster_2, min_cluster_3, min_cluster_4, to_fit, centers, c10_df, c11_df, c12_df, c13_df, c14_df)


def prediciton (min_cluster_0, min_cluster_1, min_cluster_2, min_cluster_3, min_cluster_4, to_fit, centers, c10_df, c11_df, c12_df, c13_df, c14_df):

    min_all = min((min_cluster_0, min_cluster_1, min_cluster_2, min_cluster_3, min_cluster_4))
    # taking the one with minimum distance

    if (min_all == min_cluster_0):
        print("You belong to personality group 0 where the average ext score, est score, agr score, csn score, opn score are:- ")
        print(centers[0][50], centers[0][51], centers[0][52], centers[0][53], centers[0][54])

        print("And your average ext score, est score, agr score, csn score and opn score are:-")
        print(to_fit[50], to_fit[51], to_fit[52], to_fit[53], to_fit[54])

    elif (min_all == min_cluster_1):
        print("You belong to personality group 1 where the average ext score, est score, agr score, csn score, opn score are:- ")
        print(centers[1][50], centers[1][51], centers[1][52], centers[1][53], centers[1][54])

        print("And your average ext score, est score, agr score, csn score and opn score are:-")
        print(to_fit[50], to_fit[51], to_fit[52], to_fit[53], to_fit[54])

    elif (min_all == min_cluster_2):
        print("You belong to personality group 2 where the average ext score, est score, agr score, csn score, opn score are:- ")
        print(centers[2][50], centers[2][51], centers[2][52], centers[2][53], centers[2][54])

        print("And your average ext score, est score, agr score, csn score and opn score are:-")
        print(to_fit[50], to_fit[51], to_fit[52], to_fit[53], to_fit[54])

    elif (min_all == min_cluster_3):
        print("You belong to personality group 3 where the average ext score, est score, agr score, csn score, opn score are:- ")
        print(centers[3][50], centers[3][51], centers[3][52], centers[3][53], centers[3][54])

        print("And your average ext score, est score, agr score, csn score and opn score are:-")
        print(to_fit[50], to_fit[51], to_fit[52], to_fit[53], to_fit[54])

    elif (min_all == min_cluster_4):
        print("You belong to personality group 4 where the average ext score, est score, agr score, csn score, opn score are:- ")
        print(centers[4][50], centers[4][51], centers[4][52], centers[4][53], centers[4][54])

        print("And your average ext score, est score, agr score, csn score and opn score are:-")
        print(to_fit[50], to_fit[51], to_fit[52], to_fit[53], to_fit[54])
        
    else:
        pass

    # to print your score alongside the average of people belonging to your personality group

    graphs(min_all, min_cluster_0, min_cluster_1, min_cluster_2, min_cluster_3, min_cluster_4, c10_df, c11_df, c12_df, c13_df, c14_df)


def graphs(min_all, min_cluster_0, min_cluster_1, min_cluster_2, min_cluster_3, min_cluster_4, c10_df, c11_df, c12_df, c13_df, c14_df):
    
    print ("The range in which your personality group resides in with respect to all personality traits")

    if (min_all == min_cluster_0):

        plt.figure(figsize=[15,10])
        fft=c10_df
        n=1

        for f in fft:
            plt.subplot(5,1,n)
            plt.subplots_adjust(hspace = 0.8)
            sns.countplot(x=f,  edgecolor="black", alpha=0.7, data=c10_df)
            sns.despine()
            plt.title("Cluster 00 Score Distribution related to  : {} ".format(f))
            n=n+1

        plt.tight_layout()
        plt.show()

    elif (min_all == min_cluster_1):

        plt.figure(figsize=[15,10])
        fft=c11_df
        n=1

        for f in fft:
            plt.subplot(5,1,n)
            plt.subplots_adjust(hspace = 0.8)
            sns.countplot(x=f,  edgecolor="black", alpha=0.7, data=c11_df)
            sns.despine()
            plt.title("Cluster 01 Score Distribution related to  : {} ".format(f))
            n=n+1

        plt.tight_layout()
        plt.show()

    elif (min_all == min_cluster_2):
        
        plt.figure(figsize=[15,10])
        fft=c12_df
        n=1
        for f in fft:
            plt.subplot(5,1,n)
            plt.subplots_adjust(hspace = 0.8)
            sns.countplot(x=f,  edgecolor="black", alpha=0.7, data=c12_df)
            sns.despine()
            plt.title("Cluster 02 Score Distribution related to  : {} ".format(f))
            n=n+1
        plt.tight_layout()
        plt.show()

    elif (min_all == min_cluster_3):
        
        plt.figure(figsize=[15,10])
        fft=c13_df
        n=1

        for f in fft:
            plt.subplot(5,1,n)
            plt.subplots_adjust(hspace = 0.8)
            sns.countplot(x=f,  edgecolor="black", alpha=0.7, data=c13_df)
            sns.despine()
            plt.title("Cluster 03 Score Distribution related to  : {} ".format(f))
            n=n+1

        plt.tight_layout()
        plt.show()

    elif (min_all == min_cluster_4):
        
        plt.figure(figsize=[15,10])
        fft=c14_df
        n=1

        for f in fft:
            plt.subplot(5,1,n)
            plt.subplots_adjust(hspace = 0.8)
            sns.countplot(x=f,  edgecolor="black", alpha=0.7, data=c13_df)
            sns.despine()
            plt.title("Cluster 04 Score Distribution related to  : {} ".format(f))
            n=n+1

        plt.tight_layout()
        plt.show()
        
    else:
        pass

    # to print your personality type data in form of their rank score in a bar chart showing how many counts a certain rank has

if __name__ == '__main__':
    intro()