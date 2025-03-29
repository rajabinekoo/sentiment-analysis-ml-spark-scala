package Libs

import java.nio.file.{Files, Path}
import scala.util.control.Breaks._

object DiskStorage {
  def deleteDirectoryRecursively(path: Path): Unit = {
    if (Files.isDirectory(path)) {
      val stream = Files.list(path)
      try {
        val iterator = stream.iterator()
        breakable {
          while (iterator.hasNext) {
            val entry = iterator.next()
            deleteDirectoryRecursively(entry)
          }
        }
      } finally {
        stream.close()
      }
    }
    Files.delete(path)
  }
}
