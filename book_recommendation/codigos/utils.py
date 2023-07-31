import warnings
import datetime
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
from sklearn.metrics.pairwise import cosine_similarity


def capitalize_words(string):
    try:
        words = string.split()
        capitalized_words = []
        for word in words:
            capitalized_words.append(word[0].upper() + word[1:].lower())

        return " ".join(capitalized_words)
    except:
        pass

def user_favs(id, df_filter):
    # Obtenemos los 5 libros con mejor ranking del usuario dado.
    df_users_fav = df_filter[df_filter["user-id"] == id].sort_values(["book-rating"], ascending=False)[0:5]
    return df_users_fav


def get_users_with_highest_similarity(new_df, id, df_users_pivot, df_coll_filter):

    all_users = new_df["user-id"].unique().tolist()
    if id not in all_users:
        print("❌ Usuario no encontrado ❌")
        return
    else:
        # Indice del dataframe pivotado que hace referencia al usuario dado.
        index = np.where(df_users_pivot.index == id)[0][0]
        # Calculamos la similitud entre los distintos usuarios que hay dentro del conjunto de datos.
        similarity = cosine_similarity(df_users_pivot)
        # Obtenemos una lista de la similitud que tiene el usuario dado con el resto de usuarios.
        # Lo ordenamos de mayor a menor y cogemos los 5 con mayor similitud.
        df_similarity = pd.DataFrame(similarity)
        index_similar_users = df_similarity[[index]].sort_values(index, ascending=False)[1:6].index.tolist()

        # Lista de los usuarios con mayor similitud.
        user_rec = []

        for i in index_similar_users:
            get_used_id = df_users_pivot.index[i]
            data = df_coll_filter[df_coll_filter["user-id"] == get_used_id]
            user_rec.extend(list(data.drop_duplicates("user-id")["user-id"].values))

        return user_rec


def get_recomendation_collaborative_filtering(new_df, most_similar_users, books_reading_user_id, n_books=5):

    recommend_books = []
    for i in most_similar_users:
        # Cogemos el top 5 libros más valorados de cada uno de los usuarios con mayor similitud al usuario.
        y = new_df[(new_df["user-id"] == i)]
        books = y.loc[~y["book-title"].isin(books_reading_user_id),:]
        books = books.sort_values(["book-rating"] ,ascending=False)[0:5]
        recommend_books.extend(books["book-title"].values)

    recommend_books = list(set(recommend_books))
    return recommend_books[0:n_books]


def get_info_book(df_books, listbooks):

    cols = ['book-title', 'book-author', 'publisher', 'year-of-publication']
    df_books_ = df_books[df_books['book-title'].isin(listbooks)][cols].copy()
    df_books_['year-of-publication'] = df_books_['year-of-publication']\
        .map(lambda x: int(datetime.datetime.now().year) if x == '0' else int(x))

    if len(df_books_) != len(listbooks):
        df_books_['ranking'] = df_books_.groupby('book-title')['year-of-publication']\
            .rank(method='first', ascending=True)
        return df_books_[df_books_['ranking'] == 1].drop(['ranking'], axis=1).reset_index(drop=True)
    else:
        return df_books_.reset_index(drop=True)


def recommendation_based_content(df_common_books_pivot, df_based_content, bookTitle, n_books=10, n_tops=5):

    # Calculamos la correlación entre las columnas del dataset title (que contiene la puntuación de todos los usuarios
    # que han leído el libro booktitle) vs el dataset df_common_books_pivot el cual contiene las puntuaciones de los
    # libros leídos por cada usuario. De esta manera obtenemos los libros más correlacionados con el libro dado.
    title = df_common_books_pivot[bookTitle]
    df_corr = pd.DataFrame(df_common_books_pivot.corrwith(title).sort_values(ascending=False)).reset_index(drop=False)
    # Si el propio libro este en la lista de recomendados se elimina y no se tiene en cuenta en la recomendación.
    if bookTitle in [title for title in df_corr["book-title"]]:
        df_corr = df_corr.drop(df_corr[df_corr["book-title"] == bookTitle].index[0])

    df_corr.columns = ["book-title", "correlation"]

    # Obtenemos la nota promedio de los libros que más correlación tienen con el libro indicado en "bookTitle".
    rating_recommendation_books = df_based_content[df_based_content["book-title"].isin(df_corr["book-title"])]\
        .groupby('book-title', as_index=False)["book-rating"].mean()
    df_corr = pd.merge(df_corr, rating_recommendation_books, on='book-title', how='inner')
    # Dentro de los 10 libros con mayor correlación, cogemos los 5 libros que mayor rating tengan.
    df_recommendation = df_corr[0:n_books].sort_values('book-rating', ascending=False).reset_index(drop=True)[0:n_tops]
    recommended_books = df_recommendation['book-title'].tolist()

    # Agregamos información a la recomendación añadiendo información sobre el autor y fecha de publicación.
    cols = ['book-title', 'book-author', 'antiquity']
    df_info_book = df_based_content[df_based_content['book-title'].isin(recommended_books)][cols]\
        .drop_duplicates().copy()
    df_info_book['position'] = df_info_book.groupby('book-title')['antiquity'].rank(ascending=False, method='first')
    df_info_book['year-of-publication'] = datetime.datetime.now().year - df_info_book['antiquity']
    df_info_book = df_info_book[df_info_book['position'] == 1]\
        .drop(['position', 'antiquity'], axis=1).reset_index(drop=True)
    df_recommendation = pd.merge(df_recommendation, df_info_book, on='book-title', how='left')\
        .drop(['correlation'], axis=1)

    return df_recommendation
