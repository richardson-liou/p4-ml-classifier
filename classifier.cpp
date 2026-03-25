#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <set>
#include <map>
#include "csvstream.hpp"
using namespace std;

class Classifier{
private: 

    map<string,int> label_count;
    map<string,int> post_word_count;
    map<string,map<string,int>> label_word_count;
    set<string> vocabulary;
    int total_posts = 0;

    map<string,double> log_prior;
    map<string,map<string,double>> log_likelihood;

public:
    void train(const string &train_filename);
    set<string> find_unique_words(const string &str);
};

set<string> Classifier::find_unique_words(const string &str) {
    istringstream source(str);
    set<string> words;
    string word;
    while (source >> word) {
        words.insert(word);
    }
    return words;
}

void Classifier::train(const string &train_filename) {
    csvstream train_file(train_filename);

    map<string, string> row;

    cout << "training data:" << endl;

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
}

int main(int argc, char *argv[]) {
  cout.precision(3);
  if (argc != 2 && argc != 3) {
        cout << "Usage: classifier.exe TRAIN_FILE [TEST_FILE]" << endl;
        return 1;
    }

    string train_filename = argv[1];
    string test_filename = "";
    if (argc == 3) {
        test_filename = argv[2];
    }

    Classifier clf;
    clf.train(train_filename);

    if (!test_filename.empty()) {
        // TODO: implement clf.predict_file(test_filename);
    }
    
  return 0;
}

