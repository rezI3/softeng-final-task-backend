import json

import boto3


class RekognitionImage:
    """
    Encapsulates an Amazon Rekognition image. This class is a thin wrapper
    around parts of the Boto3 Amazon Rekognition API.
    """

    def __init__(self, image, image_name='InputImage', rekognition_client=boto3.client('rekognition')):
        """
        Initializes the image object.

        :param image: Data that defines the image, either the image bytes or
                      an Amazon S3 bucket and object key.
        :param image_name: The name of the image.
        :param rekognition_client: A Boto3 Rekognition client.
        """
        self.image = image
        self.image_name = image_name
        self.rekognition_client = rekognition_client

    def detect_faces(self):
        """
        Detects faces in the image.

        :return: The list of faces found in the image.
        """
        # try:
        request = {"Bytes": self.image}

        response = self.rekognition_client.detect_faces(
            Image=request, Attributes=['ALL']
        )

        # faces = [RekognitionFace(face) for face in response['FaceDetails']]
        # logger.info("Detected %s faces.", len(faces))
        # except ClientError:
        #     logger.exception("Couldn't detect faces in %s.", self.image_name)
        #     raise
        # else:
        return response
