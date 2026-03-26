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
    map<string,int> word_post_count;
    map<string,map<string,int>> label_word_count;
    set<string> vocabulary;
    int total_posts = 0;

    map<string,double> log_prior;
    map<string,map<string,double>> log_likelihood;

public:
    void train(const string &train_filename);
    set<string> find_unique_words(const string &str);
    string predict(const string &content);
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
            word_post_count[word]++;
            label_word_count[label][word]++;
        }
    }

    cout << "trained on " << total_posts << " examples" << endl;
    cout << "vocabulary size = " << vocabulary.size() << endl;
    cout << endl;
}

string Classifier::predict(const string &content){
    set<string> words = find_unique_words(content);

    vector<string> sorted_words(words.begin(), words.end());
    sort(sorted_words.begin(), sorted_words.end());

    string best_label = "";
    double max_log_prob = -INFINITY;

    for (const auto &pair : label_count) {
        string label = pair.first;
        int label_total = label_count[label];
        double log_prior = log(double(label_total) / total_posts);

        for (const string &word : sorted_words) {
            int word_in_label_count = label_word_count[label][word];
            int word_overall_count = word_post_count[word];
            double log_likelihood;

            if (word_in_label_count > 0) {
                log_likelihood = log(static_cast<double>(word_in_label_count) / label_total);
            } else if (word_overall_count > 0) {
                log_likelihood = log(static_cast<double>(word_overall_count) / label_total);
            } else {
                log_likelihood = log(1.0 / label_total);
            }

            cout << "Log likelihood for word '" << word << "' in label '" << label << "': " << log_likelihood << endl;

            log_prior += log_likelihood;
            cout << "Total log probability for label '" << label << "': " << log_prior << endl;
        }

        if (log_prior > max_log_prob || (log_prior == max_log_prob && label < best_label)) {
            max_log_prob = log_prior;
            best_label = label;
        }
    }

    return best_label;
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
        csvstream test_file(test_filename);
        map<string, string> row;
        
        cout << "Predictions:" << endl;

        while (test_file >> row) {
            string content = row["content"];
            string predicted_label = clf.predict(content);
            cout << "Content: " << content << endl;
            cout << "Predicted label: " << predicted_label << endl << endl;
        }
    }
    
  return 0;
}

