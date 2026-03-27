#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <set>
#include <map>
#include <cmath>
#include "csvstream.hpp"
using namespace std;

class Classifier{
private: 

    map<string,int> label_count;
    map<string,int> word_post_count;
    map<string,map<string,int>> label_word_count;
    set<string> vocabulary;
    int total_posts = 0;

    double get_log_prior(const string &label);
    double get_log_likelihood(const string &label, const string &word);
public:
    void train(const string &train_filename, bool print);
    set<string> find_unique_words(const string &str);
    pair<string, double> predict(const string &content);
    void print_parameters();
    int get_total_posts() const;
    void run_tests(const string &test_filename);
};

// EFFECTS: returns total number of posts 
int Classifier::get_total_posts() const{
    return total_posts;
}

// REQUIRES: str is a valid string
// EFFECTS: returns a set of unique words in str
set<string> Classifier::find_unique_words(const string &str) {
    istringstream source(str);
    set<string> words;
    string word;
    while (source >> word) {
        words.insert(word);
    }
    return words;
}

// REQUIRES: label is a key in label_count, total_posts > 0
// EFFECTS: returns the log prior probability of the label
double Classifier::get_log_prior(const string &label) {
    return log(static_cast<double>(label_count.at(label)) / total_posts);
}

// REQUIRES: label is a key in label_count, total_posts > 0
// EFFECTS: returns the log likelihood of word given label, using different equation
//          if word was not seen in label or not seen in training data at all
double Classifier::get_log_likelihood(const string &label, const string &word) {
    int label_total = label_count.at(label);
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

    if (word_in_label_count > 0) {
        return log(static_cast<double>(word_in_label_count) / label_total);
    } else if (word_overall_count > 0) {
        return log(static_cast<double>(word_overall_count) / total_posts);
    } else {
        return log(1.0 / total_posts);
    }
}

// REQUIRES: train() was called
// EFFECTS: prints classes, log-priors and classifier parameters
//          with counts and log-likelihoods for each label and word
void Classifier::print_parameters() {
    cout << "classes:" << endl;
    for (const auto &lc : label_count) {
        cout << "  " << lc.first << ", " << lc.second
             << " examples, log-prior = " << get_log_prior(lc.first) << endl;
    }

    cout << "classifier parameters:" << endl;
    for (const auto &lc : label_count) {
        for (const auto &wc : label_word_count.at(lc.first)) {
            cout << "  " << lc.first << ":" << wc.first
                 << ", count = " << wc.second
                 << ", log-likelihood = " 
                 << get_log_likelihood(lc.first, wc.first) << endl;
        }
    }
    cout << endl;
}

// REQUIRES: train_filename is a valid CSV file with "tag" and 
//           "content" columns
// MODIFIES: label_count, word_post_count, label_word_count, vocabulary,
//           total_posts
// EFFECTS: trains the classifier with training data and updates counts.
//          if print is true, prints training data, vocabulary size, and
//          number of examples. always prints "trained on X examples" 
void Classifier::train(const string &train_filename, bool print) {
    csvstream train_file(train_filename);

    map<string, string> row;

    if (print){
        cout << "training data:" << endl;
    }

    while (train_file >> row) {
        string label = row["tag"];
        string content = row["content"];

        if (print){ 
            cout << "  label = " << label
             << ", content = " << content << endl;
        }

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
    if (print){ 
        cout << "vocabulary size = " << vocabulary.size() << endl;
        cout << endl;
}
}

// REQUIRES: train() has been called on csv file where content comes from
// EFFECTS: returns the predicted label and its log-probability score
//          for the given content
pair <string, double> Classifier::predict(const string &content){
    set<string> words = find_unique_words(content);

    string best_label = "";
    double max_log_prob = -INFINITY;

    for (const auto &lc : label_count) {
        string label = lc.first;
        double score = get_log_prior(label);

        for (const string &word : words) {
            score += get_log_likelihood(label, word);
        }
    
    if (score > max_log_prob || (score == max_log_prob && label < best_label)) {
        max_log_prob = score;
        best_label = label;
    }
}

    return {best_label, max_log_prob};
}

// REQUIRES: train() has been called, there is a valid test_filename
//           CSV file with "tag" and "content" columns
// EFFECTS: predicts labels for each post in test_filename using predict(), 
//          prints correct label, predicted label, log-probability score,
//          and content for each post, then prints overall performance
void Classifier::run_tests(const string &test_filename) {
    csvstream test_file(test_filename);
    map<string, string> row;
    int correct = 0;
    int total = 0;

    cout << "test data:" << endl;
    while (test_file >> row) {
        string correct_label = row["tag"];
        string content = row["content"];

        pair<string, double> result = predict(content);
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

int main(int argc, char *argv[]) {
  cout.precision(3);
  if (argc != 2 && argc != 3) {
    cout << "Usage: classifier.exe TRAIN_FILE [TEST_FILE]" << endl;
    return 1;
    }

    string train_filename = argv[1];
    ifstream train_check(train_filename);
    if (!train_check.is_open()) {
        cout << "Error opening file: " << train_filename << endl;
        return 1;
    }
    train_check.close();

    string test_filename = "";
    if (argc == 3) {
        test_filename = argv[2];
    }
    if (!test_filename.empty()) {
    ifstream test_check(test_filename);
    if (!test_check.is_open()) {
        cout << "Error opening file: " << test_filename << endl;
        return 1;
    }
    test_check.close();
}   
    Classifier clf;
    if (test_filename.empty()) {
        clf.train(train_filename, true);
        clf.print_parameters();
        return 0;
    }

    clf.train(train_filename, false);
    clf.run_tests(test_filename);
    return 0;

}
