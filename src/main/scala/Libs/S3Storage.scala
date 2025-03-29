package Libs

import org.apache.commons.io.IOUtils
import scala.jdk.CollectionConverters._
import com.amazonaws.auth.BasicAWSCredentials
import java.io.{File, FileOutputStream, InputStream}
import com.amazonaws.services.s3.model.ListObjectsV2Request
import com.amazonaws.services.s3.{AmazonS3, AmazonS3ClientBuilder}
import com.amazonaws.client.builder.AwsClientBuilder.EndpointConfiguration

object S3Storage {
  private lazy val s3Client: AmazonS3 = {
    val credentials = new BasicAWSCredentials(Configs.S3AccessKey, Configs.S3SecretKey)
    AmazonS3ClientBuilder.standard()
      .withEndpointConfiguration(new EndpointConfiguration(Configs.S3URL, Configs.S3ModelBucket))
      .withCredentials(new com.amazonaws.auth.AWSStaticCredentialsProvider(credentials))
      .withPathStyleAccessEnabled(true)
      .build()
  }
  
  private def correctMinioPrefix(prefix: String): String = {
    var newPrefix = prefix
    if (newPrefix.startsWith("/")) {
      newPrefix = newPrefix.substring(1)
    }
    if (!newPrefix.endsWith("/")) {
      newPrefix += "/"
    }
    newPrefix
  }
  
  def fileDownloader(objectName: String, localFilePath: String, bucketName: String): Unit = {
    val destinationFile = new File(localFilePath)

    try {
      val s3Object = s3Client.getObject(bucketName, objectName)
      val inputStream: InputStream = s3Object.getObjectContent
      IOUtils.copy(inputStream, new FileOutputStream(destinationFile))
      inputStream.close()
    } catch {
      case e: Exception => println(s"File downloading failed: ${e.getMessage}")
    }
  }

  def downloadDirectory(bucketName: String, minioPrefix: String, localDirectoryPath: String): Unit = {
    val localDir = new File(localDirectoryPath)
    val correctedPrefix = correctMinioPrefix(minioPrefix)
    if (!localDir.exists()) {
      localDir.mkdirs()
    }

    var continuationToken: String = null
    do {
      val listObjectsRequest = new ListObjectsV2Request()
        .withBucketName(bucketName)
        .withPrefix(correctedPrefix)
        .withContinuationToken(continuationToken)

      val listObjectsResponse = s3Client.listObjectsV2(listObjectsRequest)
      
      for (s3ObjectSummary <- listObjectsResponse.getObjectSummaries.asScala) {
        val objectKey = s3ObjectSummary.getKey

        val relativePath = objectKey.substring(correctedPrefix.length)
        val localFilePath = new File(localDirectoryPath, relativePath)
        
        if (!objectKey.endsWith("/")) {
          val parentDir = localFilePath.getParentFile
          if (parentDir != null && !parentDir.exists()) {
            parentDir.mkdirs()
          }

          try {
            val s3Object = s3Client.getObject(bucketName, objectKey)
            val inputStream: InputStream = s3Object.getObjectContent
            IOUtils.copy(inputStream, new FileOutputStream(localFilePath))
            inputStream.close()
          } catch {
            case e: Exception => println(s"Downloading directory failed: ('$objectKey': ${e.getMessage})")
          }
        }
      }
      continuationToken = listObjectsResponse.getNextContinuationToken
    } while (continuationToken != null)
  }
  
  def fileUploader(localFilePath: String, bucketName: String): Unit = {
    val fileToUpload = new File(localFilePath)
    try {
      s3Client.putObject(bucketName, fileToUpload.getName, fileToUpload)
    } catch {
      case e: Exception => println(s"File uploading error: ${e.getMessage}")
    }
  }

  def uploadDirectory(localDirectoryPath: String, bucketName: String, minioPrefix: String = ""): Unit = {
    val localDirectory = new File(localDirectoryPath)
    if (!localDirectory.exists() || !localDirectory.isDirectory) {
      println("Not a directory")
      return
    }

    val files = localDirectory.listFiles()
    if (files != null) {
      for (file <- files) {
        if (file.isFile) {
          val objectKey = if (minioPrefix.isEmpty) file.getName else s"$minioPrefix/${file.getName}"
          try {
            s3Client.putObject(bucketName, objectKey, file)
          } catch {
            case e: Exception => 
              println(s"File uploading failed (Service: uploadDirectory) '${file.getAbsolutePath}': ${e.getMessage}")
          }
        } else if (file.isDirectory) {
          val newPrefix = if (minioPrefix.isEmpty) file.getName else s"$minioPrefix/${file.getName}"
          uploadDirectory(file.getAbsolutePath, bucketName, newPrefix)
        }
      }
    }
  }
}
