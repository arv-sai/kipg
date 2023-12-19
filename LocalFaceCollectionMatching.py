import pandas as pd
import boto3
from botocore.exceptions import BotoCoreError, ClientError

rekognition_client = boto3.client('rekognition', region_name='us-east-2')
s3_client = boto3.client('s3', region_name='us-east-2')
bucket_name = "photorecognitionsai"
collection_id = "FootballBaseImages_V2"


def create_collection(collection_id):
    try:
        response = rekognition_client.create_collection(
            CollectionId=collection_id)
        print(f"Collection {collection_id} created.")
        return response
    except ClientError as e:
        print(f"Client error: {e}")
        return None


def add_image_to_collection(bucket, image_name, collection_id):
    try:
        external_image_id = image_name.split('/')[-1]
        response = rekognition_client.index_faces(
            CollectionId=collection_id,
            Image={'S3Object': {'Bucket': bucket, 'Name': image_name}},
            ExternalImageId=external_image_id,
            DetectionAttributes=['ALL']
        )
        return response
    except ClientError as e:
        print(f"Client error: {e}")
        return None


def search_faces_in_collection(bucket, image_name, collection_id, threshold=70, max_faces=70):
    try:
        response = rekognition_client.search_faces_by_image(
            CollectionId=collection_id,
            Image={'S3Object': {'Bucket': bucket, 'Name': image_name}},
            FaceMatchThreshold=threshold,
            MaxFaces=max_faces
        )
        return response
    except ClientError as e:
        print(f"Client error: {e}")
        return None


def list_files(bucket, prefix):
    try:
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        return [item['Key'] for item in response.get('Contents', []) if not item['Key'].endswith('/')]
    except ClientError as e:
        print(f"An error occurred: {e}")
        return []


create_collection(collection_id)

base_images = list_files(bucket_name, 'base_football/')
for base_image in base_images:
    add_image_to_collection(bucket_name, base_image, collection_id)

game_time_images = list_files(bucket_name, 'gametime_football/')
matches_list = []

for game_image in game_time_images:
    result = search_faces_in_collection(bucket_name, game_image, collection_id)
    if result:
        face_matches = result.get('FaceMatches', [])
        if face_matches:
            for match in face_matches:
                similarity = match['Similarity']
                matched_image = match['Face']['ExternalImageId']
                print(
                    f"Match found: {matched_image} matches with {game_image} - Confidence: {similarity}%")
                matches_list.append(
                    {'Base Image': matched_image, 'Game Time Image': game_image, 'Confidence': similarity})
        else:
            print(f"No match found for {game_image}.")

matches_df = pd.DataFrame(matches_list)

matches_df.to_excel(
    '/Users/adity/Downloads/PhotoRekognitionAWS/matching_results_Football.xlsx', index=False)
