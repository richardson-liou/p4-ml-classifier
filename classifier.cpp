#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <set>
#include <map>
#include "csvstream.hpp"
using namespace std;

map<string,int> label_count;
map<string,int> post_word_count;
map<string,map<string,int>> label_word_count;
set<string> vocabulary;
int total_posts = 0;

set<string> find_unique_words(const string &str) {
    istringstream source(str);
    set<string> words;
    string word;
    while (source >> word) {
        words.insert(word);
    }
    return words;
}

int main(int argc, char *argv[]) {
  cout.precision(3);

  if (argc != 2 && argc != 3) {
      cout << "Usage: classifier.exe TRAIN_FILE [TEST_FILE]" << endl;
      return 1;
  }

  string train_filename = argv[1];

  ifstream fin(train_filename);
  if (!fin.is_open()) {
      cout << "Error opening file: " << train_filename << endl;
      return 1;
  }
  fin.close();

  csvstream train_file(train_filename);

  cout << "training data:" << endl;

  map<string, string> row;

  while (train_file >> row) {
        string label = row["tag"];
        string content = row["content"];

        cout << "  label = " << label
             << ", content = " << content << endl;

        total_posts++;
        label_count[label]++;

        set<string> words = find_unique_words(content);

        for (const string &word : words) {
            vocabulary.insert(word);
            post_word_count[word]++;
            label_word_count[label][word]++;
        }
    }

    cout << "trained on " << total_posts << " examples" << endl;
    cout << "vocabulary size = " << vocabulary.size() << endl;
    cout << endl;

  return 0;
}

