-- Host: localhost    Database: mvcdata
-- ------------------------------------------------------
-- Server version	8.0.29

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8mb4 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `admins`
--

DROP TABLE IF EXISTS `admins`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `admins` (
  `username` varchar(50) NOT NULL,
  `password` varchar(256) NOT NULL,
  PRIMARY KEY (`username`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `admins`
--

LOCK TABLES `admins` WRITE;
/*!40000 ALTER TABLE `admins` DISABLE KEYS */;
INSERT INTO `admins` VALUES ('Ankur','8eba9c09aa6ca2be93be1d8aed728be7ae3012a192137f853d2cab9e15e55765'),('Parth','4ec1a76282e4f89a809b90cea83ef509a5eda8d8d128819905a7459830a2ebe5');
/*!40000 ALTER TABLE `admins` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `books`
--

DROP TABLE IF EXISTS `books`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `books` (
  `bookid` bigint unsigned NOT NULL AUTO_INCREMENT,
  `title` varchar(50) NOT NULL,
  `author` varchar(50) DEFAULT NULL,
  `publisher` varchar(50) DEFAULT NULL,
  `numberofbooks` int DEFAULT NULL,
  PRIMARY KEY (`bookid`),
  UNIQUE KEY `bookid` (`bookid`)
) ENGINE=InnoDB AUTO_INCREMENT=15 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `books`
--

LOCK TABLES `books` WRITE;
/*!40000 ALTER TABLE `books` DISABLE KEYS */;
INSERT INTO `books` VALUES (1,'Alchemist','Paulo Coelho','Publisher',6),(3,'Kafka','Haruki Murakami','Ninetails',7),(4,'Gerome Siltson','Thea Siltson','Random guy101',0),(5,'The road not taken',NULL,'Robert Frost',2),(10,'Prize','bisht','Labs',4),(11,'Noone','Literally noone','Noone',5);
/*!40000 ALTER TABLE `books` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `checkouts`
--

DROP TABLE IF EXISTS `checkouts`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `checkouts` (
  `id` bigint unsigned NOT NULL AUTO_INCREMENT,
  `user_id` varchar(50) NOT NULL,
  `book_id` bigint unsigned NOT NULL,
  `checkout_time` timestamp NULL DEFAULT NULL,
  `checkout_adminid` varchar(50) DEFAULT NULL,
  `checkin_time` timestamp NULL DEFAULT NULL,
  `checkin_adminid` varchar(50) DEFAULT NULL,
  `fine` int DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `id` (`id`),
  KEY `user_id` (`user_id`),
  KEY `book_id` (`book_id`),
  CONSTRAINT `checkouts_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `users` (`username`) ON DELETE CASCADE,
  CONSTRAINT `checkouts_ibfk_2` FOREIGN KEY (`book_id`) REFERENCES `books` (`bookid`) ON DELETE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=36 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `checkouts`
--

LOCK TABLES `checkouts` WRITE;
/*!40000 ALTER TABLE `checkouts` DISABLE KEYS */;
INSERT INTO `checkouts` VALUES (2,'Ashish',1,'2022-07-11 21:28:29','Ankur','2022-07-11 07:39:10','Ankur',NULL),(3,'Ashpect',1,'2022-07-14 02:53:35','Ankur',NULL,NULL,NULL),(4,'Ashish',1,'2022-07-12 16:10:32','Ankur','2022-07-12 17:10:53','Ankur',NULL),(5,'Ananya',3,'2022-07-12 10:22:38','Ankur','2022-07-12 12:44:20','Ankur',NULL),(7,'Ananya',4,'2022-07-11 17:02:13','Parth','2022-07-11 17:03:58','Ankur',NULL),(8,'Ananya',1,'2022-07-11 17:10:21','Parth','2022-07-11 17:10:40','Ankur',NULL),(9,'Ananya',5,'2022-07-12 10:29:05','Ankur','2022-07-13 12:46:14','Ankur',NULL),(13,'Ananya',5,'2022-07-12 10:26:06','Ankur','2022-07-12 11:11:42','Ankur',NULL),(14,'Ananya',4,'2022-07-12 11:11:03','Ankur','2022-07-12 11:11:47','Ankur',NULL),(18,'Ashish',3,'2022-07-12 17:11:29','Ankur',NULL,NULL,NULL),(19,'Ashish',4,'2022-07-12 17:11:30','Ankur',NULL,NULL,NULL),(22,'Ashish',5,'2022-07-14 01:31:01','Ankur',NULL,NULL,NULL),(23,'Ananya',3,NULL,NULL,NULL,NULL,NULL),(24,'Ananya',4,'2022-07-13 16:11:23','Ankur','2022-07-14 02:54:51',NULL,NULL),(25,'Ashish',3,NULL,NULL,NULL,NULL,NULL),(26,'Ashish',3,NULL,NULL,NULL,NULL,NULL),(27,'Ashish',5,NULL,NULL,NULL,NULL,NULL),(28,'Ashish',5,NULL,NULL,NULL,NULL,NULL),(31,'Ananya',3,'2022-07-15 03:02:24','Ankur','2022-07-15 03:02:37',NULL,NULL),(32,'Ananya',1,'2022-07-15 03:02:19','Ankur','2022-07-15 18:31:07',NULL,NULL),(34,'Ananya',3,NULL,NULL,NULL,NULL,NULL),(35,'Ashish',4,NULL,NULL,NULL,NULL,NULL);
/*!40000 ALTER TABLE `checkouts` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `users`
--

DROP TABLE IF EXISTS `users`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `users` (
  `username` varchar(50) NOT NULL,
  `password` varchar(256) NOT NULL,
  `fullname` varchar(256) DEFAULT NULL,
  `phonenumber` bigint unsigned DEFAULT NULL,
  PRIMARY KEY (`username`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `users`
--

LOCK TABLES `users` WRITE;
/*!40000 ALTER TABLE `users` DISABLE KEYS */;
INSERT INTO `users` VALUES ('Ananya','797220df0279829c570995ba05daec77398a1076c5cade1d42ccf5d9be99d907','Aananya',987654321),('Ashish','ff81423fd44fc0fb20422ffb699ce225bacb400fb70ee3fee3743aac45e910b4','Ashishh',9458594002),('Ashpect','87562618590ab85ac97da9bbe465188761237edd67406723031f1ff4fde41063','Ashhpectt',123456789),('Avii','168aea49d2956b6e91eaa1c0427e70d51a7d8eeb6531124a8127317ac2705ca2','Aviii',987654329);
/*!40000 ALTER TABLE `users` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;