#!/usr/bin/perl
#This program computes conditional entropy from bigrams.
use v5.16.2;
use strict;
use warnings;

my $input = <STDIN>;
open (my $file, "$input");
open (my $output, ">", "conditionalEntropyBigrams.txt");

sub log2 {
    my $n = shift;
	return log($n)/log(2);
    }

my (%hash, @word, @freq, $total, $type);

foreach (<$file>){
		my ($word, $freq) = split /\t/;
		chomp $freq;
		if ($freq >0){
		if (!exists $hash{$word}){
		$hash{$word}=$freq;
		}
		else {
		$hash{$word}+=$freq;
		}
		$total+=$freq;
		}
}

my @type = keys %hash;
$type = @type;

my $most;
my $most_freq=0;
my $hapax=0;

foreach my $word (keys %hash) {
    if ($hash{$word} > $most_freq) {
      $most = $word;
      $most_freq = $hash{$word};
    }
    if ($hash{$word} == 1) {
      $hapax++;
    }
  }

print {$output} "Total frequency of bigram (token) is $total\n";
print {$output} "Total number of bigram (type) is $type\n";
print {$output} "Most frequent bigram is $most\n";
print {$output} "The frequency of the most frequent bigram is $most_freq\n";
print {$output} "The number of hapax is $hapax\n";

my (%count, @x, @y);
foreach my $word (sort keys %hash){ 
		my ($x, $y) = split /_/, $word;
		if (!exists $count{$x}){
		$count{$x}=$hash{$word};
		}
		else {
		$count{$x}+=$hash{$word};
		}
}	


foreach my $word (sort keys %hash){
		my ($x, $y) = split /_/, $word;
		if ($count{$x} >0.0){
		$hash{$word}=$hash{$word}/$count{$x};
		}
}	
		
foreach my $word (sort keys %hash){
		if ($hash{$word}>0.0){
		my $log = log2 ($hash{$word})*-1;
		$hash{$word}*=$log;
		}
}

my %sum;		
foreach my $word (sort keys %hash){
		my ($x, $y) = split /_/, $word;
		if (!exists $count{$x}){
		$sum{$x}=$hash{$word};
		}
		else {
		$sum{$x}+=$hash{$word};
		}
}


my %prob;
foreach my $x (sort keys %count){
		if (exists $count{$x}){
		$prob{$x}=$count{$x}/$total;
		}
}

my ($entropy, %ent);
foreach my $x (sort keys %prob){ 
	if (!exists $prob{$x}){
	$ent{$x}=$prob{$x}*$sum{$x};
	}
	else{
	$ent{$x}+=$prob{$x}*$sum{$x};
	}
	$entropy+=$prob{$x}*$sum{$x};
}
	
print {$output} "The value of conditional entropy is $entropy\n";

close ($file);
close ($output);		