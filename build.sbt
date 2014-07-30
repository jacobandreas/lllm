name := "lllm"

version := "1.0"
    
resolvers ++= Seq(
    Resolver.mavenLocal,
    Resolver.sonatypeRepo("snapshots"),
    Resolver.sonatypeRepo("releases"),
    Resolver.sonatypeRepo("public"),
    Resolver.typesafeRepo("releases")
)

libraryDependencies ++= Seq(
  "edu.berkeley.nlp.cs" %% "igor" % "0.1-SNAPSHOT",
  "edu.berkeley.nlp.cs" %% "erector" % "0.1-SNAPSHOT",
  "org.scalanlp" %% "breeze" % "0.9-SNAPSHOT",
  "org.apache.logging.log4j" % "log4j-slf4j-impl" % "2.0-beta9",
  "org.apache.logging.log4j" % "log4j-core" % "2.0-beta9",
  "org.apache.logging.log4j" % "log4j-api" % "2.0-beta9"
)

fork := true

javaOptions := Seq("-Xmx6g", "-Xrunhprof:cpu=samples,depth=12", "-Dlog4j.configurationFile=src/main/resources/log4j.xml")
