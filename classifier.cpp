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
    pair<string, double> predict(const string &content);
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

pair <string, double> Classifier::predict(const string &content){
    set<string> words = find_unique_words(content);
    vector<string> sorted_words(words.begin(), words.end());
    sort(sorted_words.begin(), sorted_words.end());

    string best_label = "";
    double max_log_prob = -INFINITY;

    for (const auto &lc : label_count) {
        string label = lc.first;
        int label_total = lc.second;
        double log_prior = log(static_cast<double>(label_total) / total_posts);

        for (const string &word : sorted_words) {
            int word_in_label_count = 0;
            auto label_it = label_word_count.find(label);
            if (label_it != label_word_count.end()) {
                auto word_it = label_it->second.find(word);
                if (word_it != label_it->second.end()) {
                    word_in_label_count = word_it->second;
                }
            }

            int word_overall_count = 0;
            auto overall_it = word_post_count.find(word);
            if (overall_it != word_post_count.end()) {
                word_overall_count = overall_it->second;
            }
            
            double log_likelihood;
            if (word_in_label_count > 0) {
                log_likelihood = log(static_cast<double>(word_in_label_count) / label_total);
            } else if (word_overall_count > 0) {
                log_likelihood = log(static_cast<double>(word_overall_count) / total_posts);
            } else {
                log_likelihood = log(1.0 / total_posts);
            }

            log_prior += log_likelihood;
        }

        if (log_prior > max_log_prob || (log_prior == max_log_prob && label < best_label)) {
            max_log_prob = log_prior;
            best_label = label;
        }
    }

    return {best_label, max_log_prob};
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
        int correct = 0;
        int total = 0;
        
        cout << "Predictions:" << endl;

        while (test_file >> row) {
            string correct_label = row["tag"];
            string content = row["content"];

            pair<string, double> result = clf.predict(content);
            string predicted_label = result.first;
            double log_prob = result.second;

            cout << "  correct = " << correct_label
                 << ", predicted = " << predicted_label
                 << ", log-probability score = " << log_prob << endl;
            cout << "  content = " << content << endl;
            cout << endl;

            if (predicted_label == correct_label) correct++;
            total++;
        }

        cout << "performance: " << correct << " / " << total
             << " posts predicted correctly" << endl;
    }
  return 0;
}

