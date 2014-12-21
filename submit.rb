#!/usr/bin/ruby

javac = "/System/Library/Java/JavaVirtualMachines/1.6.0.jdk/Contents/Commands/javac"

timestamp=`date +%Y-%m-%d_%H-%M-%S`.strip
work_dir="/work2/jda/lllm/work/#{timestamp}"

ssh_command_1 = <<-eof
  mkdir "#{work_dir}";
eof

ssh_command_2 = <<-eof
  cd "#{work_dir}";
  qsub ../../pbs.sh;
eof

`sbt assembly`
`tar czf src.tar.gz src`
`ssh zen.millennium.berkeley.edu "#{ssh_command_1}"`
`scp src.tar.gz zen.millennium.berkeley.edu:#{work_dir}`
`scp target/scala-2.11/lllm-assembly-0.1-SNAPSHOT.jar zen.millennium.berkeley.edu:#{work_dir}/assembly.jar`
`ssh zen.millennium.berkeley.edu "#{ssh_command_2}"`
